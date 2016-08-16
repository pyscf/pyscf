import time
import tempfile
import numpy
import numpy as np
import h5py

from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.rccsd
from pyscf.cc import uintermediates as imd

#einsum = np.einsum
einsum = pbclib.einsum

# This is unrestricted (U)CCSD, i.e. spin-orbital form.

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc()
        nvir = cc.nmo() - nocc
        t1 = numpy.zeros((nocc,nvir), eris.dtype)
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, max_memory)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    mo_e = eris.fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    tau = imd.make_tau(t2,t1,t1)

    Fvv = imd.cc_Fvv(t1,t2,eris)
    Foo = imd.cc_Foo(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)
    Woooo = imd.cc_Woooo(t1,t2,eris)
    Wvvvv = imd.cc_Wvvvv(t1,t2,eris)
    Wovvo = imd.cc_Wovvo(t1,t2,eris)

    # Move energy terms to the other side
    Fvv -= fvv
    Foo -= foo

    # T1 equation
    # TODO: Does this need a conj()? Usually zero w/ canonical HF.
    t1new = fov.copy()
    t1new +=  einsum('ie,ae->ia',t1,Fvv)
    t1new += -einsum('ma,mi->ia',t1,Foo) 
    t1new +=  einsum('imae,me->ia',t2,Fov)
    t1new += -einsum('nf,naif->ia',t1,eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia',t2,eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia',t2,eris.ooov)
    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36) 
    t2new = np.array(eris.oovv, copy=True).conj()
    Ftmp = Fvv - 0.5*einsum('mb,me->be',t1,Fov)
    tmp = einsum('ijae,be->ijab',t2,Ftmp)
    t2new += (tmp - tmp.transpose(0,1,3,2))
    Ftmp = Foo + 0.5*einsum('je,me->mj',t1,Fov)
    tmp = einsum('imab,mj->ijab',t2,Ftmp)
    t2new -= (tmp - tmp.transpose(1,0,2,3))
    t2new += 0.5*einsum('mnab,mnij->ijab',tau,Woooo) 
    for a in range(nvir):
        t2new[:,:,a,:] += 0.5*einsum('ijef,bef->ijb',tau,Wvvvv[a])
    tmp = einsum('imae,mbej->ijab',t2,Wovvo) 
    tmp -= -einsum('ie,ma,mbje->ijab',t1,t1,eris.ovov)
    t2new += ( tmp - tmp.transpose(0,1,3,2) 
             - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2) )
    tmp = einsum('ie,jeba->ijab',t1,np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,mbij->ijab',t1,eris.ovoo)
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    t1new /= eia
    t2new /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = einsum('ia,ia', fock[:nocc,nocc:], t1)
    t1t1 = einsum('ia,jb->ijab',t1,t1)
    tau = t2 + 2*t1t1
    eris_oovv = np.array(eris.oovv)
    e += 0.25 * np.dot(tau.flatten(), eris_oovv.flatten())
    #e += (0.25*np.einsum('ijab,ijab',t2,eris_oovv)
    #      + 0.5*np.einsum('ia,jb,ijab',t1,t1,eris_oovv))
    return e.real


class UCCSD(pyscf.cc.rccsd.RCCSD):

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        # TODO(TCB): Check that RCCSD init is safe on UHF input
        pyscf.cc.rccsd.RCCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)
        # Spin-orbital CCSD needs a stricter tolerance
        self.conv_tol = 1e-8
        self.conv_tol_normt = 1e-6

    def nocc(self):
        self._nocc = self._scf.mol.nelectron
        return self._nocc

    def nmo(self):
        self._nmo = self.mo_energy[0].size + self.mo_energy[1].size
        return self._nmo

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc()
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        t1 = eris.fock[:nocc,nocc:] / eia
        eris_oovv = np.array(eris.oovv)
        t2 = eris_oovv/eijab
        self.emp2 = 0.25*einsum('ijab,ijab',t2,eris_oovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def ccsd(self, t1=None, t2=None, mo_coeff=None, eris=None):
        if eris is None: eris = self.ao2mo(mo_coeff)
        self.eris = eris
        self.dump_flags()
        self._conv, self.ecc, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       max_memory=self.max_memory, verbose=self.verbose)
        if self._conv:
            logger.info(self, 'CCSD converged')
        else:
            logger.info(self, 'CCSD not converge')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.ecc)
        else:
            logger.info(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.ecc+self._scf.e_tot, self.ecc)
        return self.ecc, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris, max_memory=2000):
        return update_amps(self, t1, t2, eris, max_memory)

    def nip(self):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        self._nip = nocc + nocc*(nocc-1)/2*nvir
        return self._nip

    def nea(self):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        self._nea = nvir + nocc*nvir*(nvir-1)/2
        return self._nea

    def nee(self):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        self._nee = nocc*nvir + nocc*(nocc-1)/2*nvir*(nvir-1)/2
        return self._nee

    def ipccsd_matvec(self, vector):
        # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip()
            self.made_ip_imds = True
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

        # Eq. (8)
        Hr1 = -einsum('mi,m->i',imds.Foo,r1)
        Hr1 += einsum('me,mie->i',imds.Fov,r2)
        Hr1 += -0.5*einsum('nmie,mne->i',imds.Wooov,r2)
        # Eq. (9)
        tmp1 = einsum('mi,mja->ija',imds.Foo,r2)
        tmp2 = einsum('maei,mje->ija',imds.Wovvo,r2)
        Hr2 =  einsum('ae,ije->ija',imds.Fvv,r2)
        Hr2 += (-tmp1 + tmp1.transpose(1,0,2))
        Hr2 += -einsum('maji,m->ija',imds.Wovoo,r1)
        Hr2 += 0.5*einsum('mnij,mna->ija',imds.Woooo,r2)
        Hr2 += (tmp2 - tmp2.transpose(1,0,2))
        Hr2 += 0.5*einsum('mnef,ijae,mnf->ija',imds.Woovv,self.t2,r2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nocc].copy()
        r2 = np.zeros((nocc,nocc,nvir), vector.dtype)
        index = nocc
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    r2[i,j,a] =  vector[index]
                    r2[j,i,a] = -vector[index]
                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nocc + nocc*(nocc-1)/2*nvir
        vector = np.zeros(size, r1.dtype)
        vector[:nocc] = r1.copy()
        index = nocc
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    vector[index] = r2[i,j,a]
                    index += 1
        return vector

    def eaccsd_matvec(self,vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not self.made_ea_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ea()
            self.made_ea_imds = True
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

        # Eq. (30)
        Hr1 = einsum('ac,c->a',imds.Fvv,r1)
        Hr1 += einsum('ld,lad->a',imds.Fov,r2)
        Hr1 += 0.5*einsum('alcd,lcd->a',imds.Wvovv,r2)
        # Eq. (31)
        tmp1 = einsum('ac,jcb->jab',imds.Fvv,r2)
        tmp2 = einsum('lbdj,lad->jab',imds.Wovvo,r2)
        Hr2 = einsum('abcj,c->jab',imds.Wvvvo,r1)
        Hr2 += (tmp1 - tmp1.transpose(0,2,1))
        Hr2 += -einsum('lj,lab->jab',imds.Foo,r2)
        Hr2 += (tmp2 - tmp2.transpose(0,2,1))
        nvir = self.nmo()-self.nocc()
        for a in range(nvir):
            Hr2[:,a,:] += 0.5*einsum('bcd,jcd->jb',imds.Wvvvv[a],r2) 
        Hr2 += -0.5*einsum('klcd,lcd,kjab->jab',imds.Woovv,r2,self.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nvir].copy()
        r2 = np.zeros((nocc,nvir,nvir), vector.dtype)
        index = nvir
        for i in range(nocc):
            for a in range(nvir):
                for b in range(a):
                    r2[i,a,b] =  vector[index]
                    r2[i,b,a] = -vector[index]
                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nvir + nvir*(nvir-1)/2*nocc
        vector = np.zeros(size, r1.dtype)
        vector[:nvir] = r1.copy()
        index = nvir
        for i in range(nocc):
            for a in range(nvir):
                for b in range(a):
                    vector[index] = r2[i,a,b]
                    index += 1
        return vector

    def eeccsd_matvec(self,vector):
        # Ref: Wang, Tu, and Wang, J. Chem. Theory Comput. 10, 5567 (2014) Eqs.(9)-(10)
        # Note: Last line in Eq. (10) is superfluous.
        # See, e.g. Gwaltney, Nooijen, and Barlett, Chem. Phys. Lett. 248, 189 (1996)
        if not self.made_ee_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ee()
            self.made_ee_imds = True
        imds = self.imds

        #TODO: Check and clean-up intermediates for UCCSD

        r1,r2 = self.vector_to_amplitudes_ee(vector)

        # Additional intermediates
        Wvoov =  imds.Wovvo.transpose(1,0,3,2)
        Wovvv = -imds.Wvovv.transpose(1,0,2,3)
        Woovo = -imds.Wooov.transpose(0,1,3,2)

        # Eq. (9)
        Hr1 = einsum('ae,ie->ia',imds.Fvv,r1) 
        Hr1 += -einsum('mi,ma->ia',imds.Foo,r1) 
        Hr1 += einsum('me,imae->ia',imds.Fov,r2) 
        Hr1 += einsum('amie,me->ia',Wvoov,r1) 
        Hr1 += -0.5*einsum('mnie,mnae->ia',imds.Wooov,r2) 
        Hr1 += 0.5*einsum('amef,imef->ia',imds.Wvovv,r2)
        # Eq. (10)
        tmpab = einsum('be,ijae->ijab',imds.Fvv,r2)
        tmpab += -0.5*einsum('mnef,ijae,mnbf->ijab',imds.Woovv,self.t2,r2)
        tmpab += -einsum('mbij,ma->ijab',imds.Wovoo,r1)
        tmpab += einsum('maef,ijfb,me->ijab',Wovvv,self.t2,r1)
        tmpij = -einsum('mj,imab->ijab',imds.Foo,r2)
        tmpij += -0.5*einsum('mnef,imab,jnef->ijab',imds.Woovv,self.t2,r2)
        tmpij += einsum('abej,ie->ijab',imds.Wvvvo,r1)
        tmpij += -einsum('mnei,njab,me->ijab',Woovo,self.t2,r1)

        tmpabij = einsum('mbej,imae->ijab',imds.Wovvo,r2)

        Hr2 = ( tmpab - tmpab.transpose(0,1,3,2)
               + tmpij - tmpij.transpose(1,0,2,3)
               + 0.5*einsum('mnij,mnab->ijab',imds.Woooo,r2)
               + 0.5*einsum('abef,ijef->ijab',imds.Wvvvv,r2)
               + tmpabij - tmpabij.transpose(0,1,3,2)
               - tmpabij.transpose(1,0,2,3) + tmpabij.transpose(1,0,3,2) )

        vector = self.amplitudes_to_vector_ee(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ee(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nocc*nvir].copy().reshape((nocc,nvir))
        r2 = np.zeros((nocc,nocc,nvir,nvir), vector.dtype)
        index = nocc*nvir
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    for b in range(a):
        	        r2[i,j,a,b] =  vector[index]
        	        r2[j,i,a,b] = -vector[index]
        	        r2[i,j,b,a] = -vector[index]
        	        r2[j,i,b,a] =  vector[index]
	                index += 1
        return [r1,r2]

    def amplitudes_to_vector_ee(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nocc*nvir + nocc*(nocc-1)/2*nvir*(nvir-1)/2
        vector = np.zeros(size, r1.dtype)
        vector[:nocc*nvir] = r1.copy().reshape(nocc*nvir)
        index = nocc*nvir
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    for b in range(a):
        	        vector[index] = r2[i,j,a,b]
	                index += 1
        return vector


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore', 
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff
            self.fock = numpy.diag(np.append(cc.mo_energy[np.array(cc.mo_occ,dtype=bool)], 
                                             cc.mo_energy[np.logical_not(np.array(cc.mo_occ,dtype=bool))])).astype(mo_coeff.dtype)

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = pyscf.cc.rccsd._mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        # Convert to spin-orbitals and anti-symmetrize 
        so_coeff = np.zeros((nmo/2,nmo), dtype=mo_coeff.dtype)
        nocc_a = int(sum(cc.mo_occ[0]))
        nocc_b = int(sum(cc.mo_occ[1]))
        nvir_a = nmo/2 - nocc_a
        #nvir_b = nmo/2 - nocc_b
        spin = np.zeros(nmo, dtype=int)
        spin[:nocc_a] = 0
        spin[nocc_a:nocc] = 1
        spin[nocc:nocc+nvir_a] = 0
        spin[nocc+nvir_a:nmo] = 1
        so_coeff[:,:nocc_a] = mo_coeff[0][:,:nocc_a]
        so_coeff[:,nocc_a:nocc] = mo_coeff[1][:,:nocc_b]
        so_coeff[:,nocc:nocc+nvir_a] = mo_coeff[0][:,nocc_a:nmo/2]
        so_coeff[:,nocc+nvir_a:nmo] = mo_coeff[1][:,nocc_b:nmo/2]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory) 
            or cc.mol.incore_anyway):
            eri = ao2mofn(cc._scf.mol, (so_coeff,so_coeff,so_coeff,so_coeff), compact=0)
            if mo_coeff.dtype == np.float: eri = eri.real
            eri = eri.reshape((nmo,)*4)
            for i in range(nmo):
                for j in range(i):
                    if spin[i] != spin[j]:
                        eri[i,j,:,:] = eri[j,i,:,:] = 0.
                        eri[:,:,i,j] = eri[:,:,j,i] = 0.
            eri1 = eri - eri.transpose(0,3,2,1) 
            eri1 = eri1.transpose(0,2,1,3) 

            self.dtype = eri1.dtype
            self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
            self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
            self.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            self.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy() 
        else:
            print "*** Using HDF5 ERI storage ***"
            _tmpfile1 = tempfile.NamedTemporaryFile()
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = so_coeff[:,:nocc]
            orbv = so_coeff[:,nocc:]
            if mo_coeff.dtype == np.complex: ds_type = 'c16'
            else: ds_type = 'f8'
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), ds_type)
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), ds_type)
            self.vvvv = self.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)

            cput1 = time.clock(), time.time()
            # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
            buf = ao2mofn(cc._scf.mol, (orbo,so_coeff,orbo,so_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nocc,nmo))
            for i in range(nocc):
                for p in range(nmo):
                    if spin[i] != spin[p]:
                        buf[i,p,:,:] = 0.
                        buf[:,:,i,p] = 0.
            buf1 = buf - buf.transpose(0,3,2,1)
            buf1 = buf1.transpose(0,2,1,3) 
            cput1 = log.timer_debug1('transforming oopq', *cput1)
            self.dtype = buf1.dtype
            self.oooo[:,:,:,:] = buf1[:,:,:nocc,:nocc]
            self.ooov[:,:,:,:] = buf1[:,:,:nocc,nocc:]
            self.oovv[:,:,:,:] = buf1[:,:,nocc:,nocc:]

            cput1 = time.clock(), time.time()
            # <ia||pq> = <ia|pq> - <ia|qp> = (ip|aq) - (iq|ap)
            buf = ao2mofn(cc._scf.mol, (orbo,so_coeff,orbv,so_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nvir,nmo))
            for p in range(nmo):
                for i in range(nocc):
                    if spin[i] != spin[p]:
                        buf[i,p,:,:] = 0.
                for a in range(nvir):
                    if spin[nocc+a] != spin[p]:
                        buf[:,:,a,p] = 0.
            buf1 = buf - buf.transpose(0,3,2,1)
            buf1 = buf1.transpose(0,2,1,3) 
            cput1 = log.timer_debug1('transforming ovpq', *cput1)
            self.ovoo[:,:,:,:] = buf1[:,:,:nocc,:nocc]
            self.ovov[:,:,:,:] = buf1[:,:,:nocc,nocc:]
            self.ovvv[:,:,:,:] = buf1[:,:,nocc:,nocc:]

            for a in range(nvir):
                orbva = orbv[:,a].reshape(-1,1)
                buf = ao2mofn(cc._scf.mol, (orbva,orbv,orbv,orbv), compact=0)
                if mo_coeff.dtype == np.float: buf = buf.real
                buf = buf.reshape((1,nvir,nvir,nvir))
                for b in range(nvir):
                    if spin[nocc+a] != spin[nocc+b]:
                        buf[0,b,:,:] = 0.
                    for c in range(nvir):
                        if spin[nocc+b] != spin[nocc+c]:
                            buf[:,:,b,c] = buf[:,:,c,b] = 0.
                buf1 = buf - buf.transpose(0,3,2,1) 
                buf1 = buf1.transpose(0,2,1,3) 
                self.vvvv[a] = buf1[:]

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)


class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> uintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc):
        self.cc = cc
        self._made_shared = False

    def _make_shared(self):
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris
        self.Foo = imd.Foo(t1,t2,eris)
        self.Fvv = imd.Fvv(t1,t2,eris)
        self.Fov = imd.Fov(t1,t2,eris)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.oovv

    def make_ip(self):
        if self._made_shared is False:
            self._make_shared()
            self._made_shared = True

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)

    def make_ea(self):
        if self._made_shared is False:
            self._make_shared()
            self._made_shared = True

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        self.Wvvvv = imd.Wvvvv(t1,t2,eris)

    def make_ee(self):
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        #TODO(TCB): Clean up for spin-orbital CCSD

        self.Fvv = imd.Fvv(t1,t2,eris)
        self.Foo = imd.Foo(t1,t2,eris)
        self.Fov = imd.Fov(t1,t2,eris)

        self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wvvvv = imd.Wvvvv(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.Wvvvo = imd.Wvvvo(t1,t2,eris)

        # Additional intermediates
        self.Wvoov =  self.Wovvo.transpose(1,0,3,2)
        self.Wovvv = -self.Wvovv.transpose(1,0,2,3)
        self.Woovo = -self.Wooov.transpose(0,1,3,2)
        self.Woovv = eris.oovv

