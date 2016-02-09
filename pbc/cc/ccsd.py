import time
import tempfile
import numpy
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.cc.ccsd import _cp
import pyscf.pbc.ao2mo
from pyscf.pbc.cc import intermediates as imd
from pyscf.pbc.lib.linalg_helper import eigs 

#einsum = np.einsum
einsum = pbclib.einsum

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
        t1 = numpy.zeros((nocc,nvir), np.complex128)
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
    #nov = nocc*nvir
    fock = eris.fock
    #t1new = numpy.zeros_like(t1)
    #t2new = numpy.zeros_like(t2)

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    mo_e = eris.fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    tau = imd.make_tau(t2,t1,t1)

    ### From eom-cc hackathon code ###
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Foo = imd.cc_Foo(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)
    Woooo = imd.cc_Woooo(t1,t2,eris)
    Wvvvv = imd.cc_Wvvvv(t1,t2,eris)
    Wovvo = imd.cc_Wovvo(t1,t2,eris)

    # Move energy terms to the other side
    Fvv -= fvv
    Foo -= foo

    eris_oovo = - _cp(eris.ooov).transpose(0,1,3,2)
    eris_ovvo = - _cp(eris.ovov).transpose(0,1,3,2)
    eris_vvvo =   _cp(eris.ovvv).transpose(2,3,1,0).conj()

    # T1 equation
    # TODO: Does this need a conj()? Usually zero w/ canonical HF.
    t1new = _cp(fov)
    t1new +=   einsum('ie,ae->ia',t1,Fvv)
    t1new += - einsum('ma,mi->ia',t1,Foo) 
    t1new +=   einsum('imae,me->ia',t2,Fov)
    t1new += - einsum('nf,naif->ia',t1,eris.ovov)
    t1new += - 0.5*einsum('imef,maef->ia',t2,eris.ovvv)
    t1new += - 0.5*einsum('mnae,nmei->ia',t2,eris_oovo)
    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36) 
    t2new = _cp(eris.oovv).conj()
    Ftmp = Fvv - 0.5*einsum('mb,me->be',t1,Fov)
    tmp = einsum('ijae,be->ijab',t2,Ftmp)
    t2new += (tmp - tmp.transpose(0,1,3,2))
    Ftmp = Foo + 0.5*einsum('je,me->mj',t1,Fov)
    tmp = einsum('imab,mj->ijab',t2,Ftmp)
    t2new -= (tmp - tmp.transpose(1,0,2,3))
    t2new += ( 0.5*einsum('mnab,mnij->ijab',tau,Woooo) 
             + 0.5*einsum('ijef,abef->ijab',tau,Wvvvv) )
    tmp = einsum('imae,mbej->ijab',t2,Wovvo) 
    tmp -= einsum('ie,ma,mbej->ijab',t1,t1,eris_ovvo)
    t2new += ( tmp - tmp.transpose(0,1,3,2) 
             - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2) )
    tmp = einsum('ie,abej->ijab',t1,eris_vvvo)
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
    eris_oovv = _cp(eris.oovv)
    e = einsum('ia,ia', fock[:nocc,nocc:], t1)
    t1t1 = einsum('ia,jb->ijab',t1,t1)
    tau = t2 + 2*t1t1
    e += 0.25 * np.dot(tau.flatten(), eris_oovv.flatten())
    #e += (0.25*einsum('ijab,ijab',t2,eris_oovv)
    #      + 0.5*einsum('ia,jb,ijab',t1,t1,eris_oovv))
    return e.real


class CCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        nso = 2*mf.mo_energy.size
        if mo_energy is None: 
            mo_energy = np.zeros(nso)
            mo_energy[0::2] = mo_energy[1::2] = mf.mo_energy
        if mo_coeff is None:
            # TODO: Careful for real/complex here, in the future
            #mo_coeffT = mf.mo_coeff.T.real
            mo_coeffT = mf.mo_coeff.T
            so_coeffT = np.zeros((nso,nso), dtype=mo_coeffT.dtype)
            for i in range(nso):
                if i%2 == 0:
                    so_coeffT[i][:nso/2] = mo_coeffT[i//2]
                else:
                    so_coeffT[i][nso/2:] = mo_coeffT[i//2]
            # Each col is an eigenvector, first n/2 rows are alpha, then n/2 beta
            mo_coeff = so_coeffT.T
        if mo_occ is None: 
            mo_occ = np.zeros(nso)
            mo_occ[0:mf.cell.nelectron] = 1

        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)

    def nocc(self):
        # Spin orbitals
        self._nocc = 2*pyscf.cc.ccsd.CCSD.nocc(self)
        return self._nocc

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc()
        nvir = mo_e.size - nocc
        t1 = np.zeros((nocc,nvir), eris.dtype)
        #eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        #t1 = eris.fock[:nocc,nocc:] / eia
        t2 = np.zeros((nocc,nocc,nvir,nvir), eris.dtype)
        self.emp2 = 0
        foo = eris.fock[:nocc,:nocc]
        fvv = eris.fock[nocc:,nocc:]
        eris_oovv = _cp(eris.oovv)
        eia = np.zeros((nocc,nvir))
        eijab = np.zeros((nocc,nocc,nvir,nvir))
        for i in range(nocc):
            for a in range(nvir):
                eia[i,a] = (foo[i,i] - fvv[a,a]).real
                for j in range(nocc):
                    for b in range(nvir):
                        eijab[i,j,a,b] = ( foo[i,i] + foo[j,j]
                                         - fvv[a,a] - fvv[b,b] ).real
                        t2[i,j,a,b] = eris_oovv[i,j,a,b]/eijab[i,j,a,b]
        self.emp2 = 0.25*einsum('ijab,ijab',t2,eris_oovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        print "MP2 energy =", self.emp2
        return self.emp2, t1, t2

    def ccsd(self, t1=None, t2=None, mo_coeff=None, eris=None):
        if eris is None: eris = self.ao2mo(mo_coeff)
        self.eris = eris
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

    # EOM CCSD starts here

    def eeccsd(self, nroots=6):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nocc*nvir+nocc*(nocc-1)/2*nvir*(nvir-1)/2
        evals, evecs = eigs(self.eeccsd_matvec, size, 2*nroots)
        return evals.real[:nroots], evecs

    #@profile
    def eeccsd_matvec(self,vector):
    ########################################################
    # FOLLOWING:                                           #
    # Z. Wang, Z. Tu, and F. Wang                          #
    # J. Chem. Theory Comput. 10, 5567 (2014) Eqs.(9)-(10) #
    # BUT THERE IS A TYPO.                                 #
    # -- Last line in Eq. (10) is superfluous.             #
    # -- See, e.g. Gwaltney, Nooijen, and Barlett          #
    # --           Chem. Phys. Lett. 248, 189 (1996)       #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ee(vector)

        t1,t2,eris = self.t1, self.t2, self.eris
        Fvv = imd.Fvv(t1,t2,eris)
        Foo = imd.Foo(t1,t2,eris)
        Fov = imd.Fov(t1,t2,eris)

        Woooo = imd.Woooo(t1,t2,eris)
        Wvvvv = imd.Wvvvv(t1,t2,eris)
        Wovvo = imd.Wovvo(t1,t2,eris)
        Wooov = imd.Wooov(t1,t2,eris)
        Wvovv = imd.Wvovv(t1,t2,eris)
        Wovoo = imd.Wovoo(t1,t2,eris)
        Wvvvo = imd.Wvvvo(t1,t2,eris)
        # Additional intermediates
        Woovv = _cp(eris.oovv)
        Wvoov =  Wovvo.transpose(1,0,3,2)
        Wovvv = -Wvovv.transpose(1,0,2,3)
        Woovo = -Wooov.transpose(0,1,3,2)

        # Eq. (9)
        Hr1 = ( einsum('ae,ie->ia',Fvv,r1) 
               - einsum('mi,ma->ia',Foo,r1) 
               + einsum('me,imae->ia',Fov,r2) 
               + einsum('amie,me->ia',Wvoov,r1) 
               - 0.5*einsum('mnie,mnae->ia',Wooov,r2) 
               + 0.5*einsum('amef,imef->ia',Wvovv,r2) )
        # Eq. (10)
        tmpab = ( einsum('be,ijae->ijab',Fvv,r2)
                 - 0.5*einsum('mnef,ijae,mnbf->ijab',Woovv,self.t2,r2)
                 - einsum('mbij,ma->ijab',Wovoo,r1)
                 + einsum('maef,ijfb,me->ijab',Wovvv,self.t2,r1) )
        tmpij = (-einsum('mj,imab->ijab',Foo,r2)
                 - 0.5*einsum('mnef,imab,jnef->ijab',Woovv,self.t2,r2)
                 + einsum('abej,ie->ijab',Wvvvo,r1)
                 - einsum('mnei,njab,me->ijab',Woovo,self.t2,r1) )

        tmpabij = einsum('mbej,imae->ijab',Wovvo,r2)

        Hr2 = ( tmpab - tmpab.transpose(0,1,3,2)
               + tmpij - tmpij.transpose(1,0,2,3)
               + 0.5*einsum('mnij,mnab->ijab',Woooo,r2)
               + 0.5*einsum('abef,ijef->ijab',Wvvvv,r2)
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
        size = ( nocc*nvir
                 + nocc*(nocc-1)/2*nvir*(nvir-1)/2 )
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

    def ipccsd(self, nroots=6):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nocc + nocc*(nocc-1)/2*nvir
        evals, evecs = eigs(self.ipccsd_matvec, size, 2*nroots)
        return evals.real[:nroots], evecs

    #@profile
    def ipccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # Z. Tu, F. Wang, and X. Li                            #
    # J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ip(vector)

        t1,t2,eris = self.t1, self.t2, self.eris
        Fvv = imd.Fvv(t1,t2,eris)
        Foo = imd.Foo(t1,t2,eris)
        Fov = imd.Fov(t1,t2,eris)
        Woovo = -imd.Wooov(t1,t2,eris).transpose(0,1,3,2)
        Wvoov = imd.Wovvo(t1,t2,eris).transpose(1,0,3,2)
        Wovoo = imd.Wovoo(t1,t2,eris)
        Woooo = imd.Woooo(t1,t2,eris)
        Woovv = eris.oovv
        #Woovv = _cp(eris.oovv)

        # Eq. (8)
        Hr1 = (- einsum('mi,m->i',Foo,r1)
               + einsum('me,mie->i',Fov,r2)
               - 0.5*einsum('mnei,mne->i',Woovo,r2) )
        # Eq. (9)
        tmp1 = einsum('mi,mja->ija',Foo,r2)
        tmp2 = einsum('amie,mje->ija',Wvoov,r2)
        Hr2 = ( einsum('ae,ije->ija',Fvv,r2)
               - tmp1 + tmp1.transpose(1,0,2)
               - einsum('maji,m->ija',Wovoo,r1)
               + 0.5*einsum('mnij,mna->ija',Woooo,r2)
               + tmp2 - tmp2.transpose(1,0,2)
               + 0.5*einsum('mnef,ijae,mnf->ija',Woovv,self.t2,r2) )

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
        size = nocc+nocc*(nocc-1)/2*nvir
        vector = np.zeros(size, r1.dtype)
        vector[:nocc] = r1.copy()
        index = nocc
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    vector[index] = r2[i,j,a]
                    index += 1
        return vector

    def eaccsd(self,nroots):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nvir + nvir*(nvir-1)/2*nocc
        evals, evecs = eigs(self.eaccsd_matvec, size, 2*nroots)
        return evals.real[:nroots], evecs

    #@profile
    def eaccsd_matvec(self,vector):
    ########################################################
    # FOLLOWING:                                           #
    # M. Nooijen and R. J. Bartlett,                       #
    # J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ea(vector)

        t1,t2,eris = self.t1, self.t2, self.eris
        Fvv = imd.Fvv(t1,t2,eris)
        Foo = imd.Foo(t1,t2,eris)
        Fov = imd.Fov(t1,t2,eris)
        Wvovv = imd.Wvovv(t1,t2,eris)
        Wvvvo = imd.Wvvvo(t1,t2,eris)
        Wovvo = imd.Wovvo(t1,t2,eris)
        Wvvvv = imd.Wvvvv(t1,t2,eris)
        #Woovv = _cp(eris.oovv)
        Woovv = eris.oovv

        # Eq. (30)
        Hr1 = ( einsum('ac,c->a',Fvv,r1)
               + einsum('ld,lad->a',Fov,r2)
               + 0.5*einsum('alcd,lcd->a',Wvovv,r2) )

        # Eq. (31)
        tmp1 = einsum('ac,jcb->jab',Fvv,r2)
        tmp2 = einsum('lbdj,lad->jab',Wovvo,r2)
        Hr2 = ( einsum('abcj,c->jab',Wvvvo,r1)
               + tmp1 - tmp1.transpose(0,2,1)
               - einsum('lj,lab->jab',Foo,r2)
               + tmp2 - tmp2.transpose(0,2,1)
               + 0.5*einsum('abcd,jcd->jab',Wvvvv,r2) 
               - 0.5*einsum('klcd,lcd,kjab->jab',Woovv,r2,t2) )

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
        size = nvir+nvir*(nvir-1)/2*nocc
        vector = np.zeros(size, r1.dtype)
        vector[:nvir] = r1.copy()
        index = nvir
        for i in range(nocc):
            for a in range(nvir):
                for b in range(a):
                    vector[index] = r2[i,a,b]
                    index += 1
        return vector


class _ERIS:
    """_ERIS handler for PBCs."""
    def __init__(self, cc, mo_coeff=None, method='incore'):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(cc.mo_energy.size, dtype=numpy.bool)
        if isinstance(cc.frozen, (int, numpy.integer)):
            moidx[:cc.frozen] = False
        elif len(cc.frozen) > 0:
            moidx[numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff[:,moidx]
            self.fock = numpy.diag(cc.mo_energy[moidx]).astype(mo_coeff.dtype)
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = pyscf.cc.ccsd._mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        # Convert to spin-orbitals and anti-symmetrize 
        so_coeff = np.zeros((nmo/2,nmo), dtype=mo_coeff.dtype)
        so_coeff[:,::2] = so_coeff[:,1::2] = mo_coeff[:nmo/2,::2]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and cc._scf._eri is not None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):

            eri = pyscf.pbc.ao2mo.general(cc._scf.cell, (so_coeff,so_coeff,
                                                         so_coeff,so_coeff))
                                                        #so_coeff,so_coeff)).real
            eri = eri.reshape((nmo,)*4)
            eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0.
            #print "ERI ="
            #print eri
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
            #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)

        else:
            print "*** Using HDF5 ERI storage ***"
            _tmpfile1 = tempfile.NamedTemporaryFile()
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = so_coeff[:,:nocc]
            orbv = so_coeff[:,nocc:]
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'c16')
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'c16')
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'c16')
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'c16')
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'c16')
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), 'c16')

            cput1 = time.clock(), time.time()
            buf = pyscf.pbc.ao2mo.general(cc._scf.cell, (orbo,so_coeff,so_coeff,so_coeff))
            buf = buf.reshape((nocc,nmo,nmo,nmo))
            buf[::2,1::2] = buf[1::2,::2] = buf[:,:,::2,1::2] = buf[:,:,1::2,::2] = 0.
            buf1 = buf - buf.transpose(0,3,2,1) 
            buf1 = buf1.transpose(0,2,1,3) 
            cput1 = log.timer_debug1('transforming oppp', *cput1)

            self.dtype = buf1.dtype
            self.oooo[:,:,:,:] = buf1[:,:nocc,:nocc,:nocc]
            self.ooov[:,:,:,:] = buf1[:,:nocc,:nocc,nocc:]
            self.ovoo[:,:,:,:] = buf1[:,nocc:,:nocc,:nocc]
            self.oovv[:,:,:,:] = buf1[:,:nocc,nocc:,nocc:]
            self.ovov[:,:,:,:] = buf1[:,nocc:,:nocc,nocc:]
            self.ovvv[:,:,:,:] = buf1[:,nocc:,nocc:,nocc:]

            self.vvvv = self.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), 'c16')
            for a in range(nvir):
                orbva = orbv[:,a].reshape(-1,1)
                buf = pyscf.pbc.ao2mo.general(cc._scf.cell, (orbva,orbv,orbv,orbv))
                buf = buf.reshape((1,nvir,nvir,nvir))
                if a%2 == 0:
                    buf[0,1::2,:,:] = 0.
                else:
                    buf[0,0::2,:,:] = 0.
                buf[:,:,::2,1::2] = buf[:,:,1::2,::2] = 0.
                buf1 = buf - buf.transpose(0,3,2,1) 
                buf1 = buf1.transpose(0,2,1,3) 
                self.vvvv[a] = buf1[:]

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)
