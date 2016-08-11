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
import pyscf.cc.ccsd
from pyscf.cc import rintermediates as imd
from pyscf.pbc.lib.linalg_helper import eigs

#einsum = np.einsum
einsum = pbclib.einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

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

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)
    Loo = imd.Loo(t1,t2,eris)
    Lvv = imd.Lvv(t1,t2,eris)
    Woooo = imd.cc_Woooo(t1,t2,eris)
    Wvvvv = imd.cc_Wvvvv(t1,t2,eris)
    Wvoov = imd.cc_Wvoov(t1,t2,eris)
    Wvovo = imd.cc_Wvovo(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= foo
    Fvv -= fvv
    Loo -= foo
    Lvv -= fvv

    # T1 equation
    # TODO: Check this conj(). Hirata and Bartlett has
    # f_{vo}(a,i), which should be equal to f_{ov}^*(i,a)
    t1new = fov.conj().copy()
    t1new += -2*einsum('kc,ka,ic->ia',fov,t1,t1)
    t1new +=   einsum('ac,ic->ia',Fvv,t1)
    t1new +=  -einsum('ki,ka->ia',Foo,t1)
    t1new +=   einsum('kc,kica->ia',Fov,2*t2)
    t1new +=   einsum('kc,ikca->ia',Fov, -t2)
    t1new +=   einsum('kc,ic,ka->ia',Fov,t1,t1)
    t1new += 2*einsum('akic,kc->ia',eris.voov,t1)
    #t1new +=  -einsum('akci,kc->ia',eris.vovo,t1)
    t1new +=  -einsum('kaic,kc->ia',eris.ovov,t1)
    t1new += 2*einsum('akcd,ikcd->ia',eris.vovv,t2)
    t1new +=  -einsum('akdc,ikcd->ia',eris.vovv,t2)
    t1new += 2*einsum('akcd,ic,kd->ia',eris.vovv,t1,t1)
    t1new +=  -einsum('akdc,ic,kd->ia',eris.vovv,t1,t1)
    t1new += -2*einsum('klic,klac->ia',eris.ooov,t2)
    t1new +=  einsum('lkic,klac->ia',eris.ooov,t2)
    t1new += -2*einsum('klic,ka,lc->ia',eris.ooov,t1,t1)
    t1new +=  einsum('lkic,ka,lc->ia',eris.ooov,t1,t1)

    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36)
    t2new = np.array(eris.oovv).conj()
    t2new += einsum('klij,klab->ijab',Woooo,t2)
    t2new += einsum('klij,ka,lb->ijab',Woooo,t1,t1)
    #t2new += einsum('abcd,ijcd->ijab',Wvvvv,t2)
    #t2new += einsum('abcd,ic,jd->ijab',Wvvvv,t1,t1)
    for a in range(nvir):
        Wvvvv_a = np.array(Wvvvv[a])
        t2new[:,:,a,:] += einsum('bcd,ijcd->ijb',Wvvvv_a,t2)
        t2new[:,:,a,:] += einsum('bcd,ic,jd->ijb',Wvvvv_a,t1,t1)
    tmp = einsum('ac,ijcb->ijab',Lvv,t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('ki,kjab->ijab',Loo,t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    #tmp2 = eris.vvov - einsum('kbic,ka->abic',eris.ovov,t1)
    tmp2 = np.array(eris.vovv).transpose(3,2,1,0).conj() \
            - einsum('kbic,ka->abic',eris.ovov,t1)
    tmp = einsum('abic,jc->ijab',tmp2,t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    #tmp2 = eris.vooo + einsum('akic,jc->akij',eris.voov,t1)
    tmp2 = np.array(eris.ooov).transpose(3,2,1,0).conj() \
            + einsum('akic,jc->akij',eris.voov,t1)
    tmp = einsum('akij,kb->ijab',tmp2,t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = 2*einsum('akic,kjcb->ijab',Wvoov,t2) - einsum('akci,kjcb->ijab',Wvovo,t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('akic,kjbc->ijab',Wvoov,t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('bkci,kjac->ijab',Wvovo,t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    t1new /= eia
    t2new /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*einsum('ia,ia', fock[:nocc,nocc:], t1)
    t1t1 = einsum('ia,jb->ijab',t1,t1)
    tau = t2 + t1t1
    e += einsum('ijab,ijab', 2*tau, eris.oovv)
    e += einsum('ijab,ijba',  -tau, eris.oovv)
    return e.real


class RCCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc()
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        eris_oovv = np.array(eris.oovv)
        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = eris_oovv/eijab
        woovv = 2*eris_oovv - eris_oovv.transpose(0,1,3,2)
        self.emp2 = einsum('ijab,ijab',t2,woovv.conj()).real
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
        self._nip = nocc + nocc*nocc*nvir
        return self._nip

    def nea(self):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        self._nea = nvir + nocc*nvir*nvir
        return self._nea

    def nee(self):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        self._nee = nocc*nvir + nocc*nocc*nvir*nvir
        return self._nee

    def ipccsd(self, nroots=1):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nip()
        nroots = min(nroots,size)
        self._ipconv, self.eip, evecs = eigs(self.ipccsd_matvec, size, nroots, verbose=log)
        if self._ipconv:
            logger.info(self, 'IP-CCSD converged')
        else:
            logger.info(self, 'IP-CCSD not converge')
        for n in range(nroots):
            logger.info(self, 'root %d E(IP-CCSD) = %.16g', n, self.eip.real[n])
        log.timer('IP-CCSD', *cput0)
        return self.eip.real[:nroots], evecs

    def ipccsd_matvec(self, vector):
        # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip()
            self.made_ip_imds = True
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

        Hr1 = -einsum('ki,k->i',imds.Loo,r1)
        Hr1 += 2*einsum('ld,ild->i',imds.Fov,r2)
        Hr1 +=  -einsum('kd,kid->i',imds.Fov,r2)
        Hr1 += -2*einsum('klid,kld->i',imds.Wooov,r2)
        Hr1 +=    einsum('lkid,kld->i',imds.Wooov,r2)

        tmp = 2*einsum('lkdc,kld->c',imds.Woovv,r2)
        tmp += -einsum('kldc,kld->c',imds.Woovv,r2)
        Hr2 = einsum('bd,ijd->ijb',imds.Lvv,r2)
        Hr2 += -einsum('ki,kjb->ijb',imds.Loo,r2)
        Hr2 += -einsum('lj,ilb->ijb',imds.Loo,r2)
        Hr2 += -einsum('kbij,k->ijb',imds.Wovoo,r1)
        Hr2 +=  einsum('klij,klb->ijb',imds.Woooo,r2)
        Hr2 += 2*einsum('lbdj,ild->ijb',imds.Wovvo,r2)
        Hr2 +=  -einsum('kbdj,kid->ijb',imds.Wovvo,r2)
        Hr2 +=  -einsum('lbjd,ild->ijb',imds.Wovov,r2) #typo in Nooijen's paper
        Hr2 +=  -einsum('kbid,kjd->ijb',imds.Wovov,r2)
        Hr2 += -einsum('c,ijcb->ijb',tmp,self.t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = self.nip()
        vector = np.zeros(size, r1.dtype)
        vector[:nocc] = r1.copy()
        vector[nocc:] = r2.copy().reshape(nocc*nocc*nvir)
        return vector

    def eaccsd(self, nroots=1):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nea()
        nroots = min(nroots,size)
        self._eaconv, self.eea, evecs = eigs(self.eaccsd_matvec, size, nroots, verbose=log)
        if self._eaconv:
            logger.info(self, 'EA-CCSD converged')
        else:
            logger.info(self, 'EA-CCSD not converge')
        for n in range(nroots):
            logger.info(self, 'root %d E(EA-CCSD) = %.16g', n, self.eea.real[n])
        log.timer('EA-CCSD', *cput0)
        return self.eea.real[:nroots], evecs

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
        Hr1 =  einsum('ac,c->a',imds.Lvv,r1)
        Hr1 += einsum('ld,lad->a',2.*imds.Fov,r2)
        Hr1 += einsum('ld,lda->a',  -imds.Fov,r2)
        Hr1 += 2*einsum('alcd,lcd->a',imds.Wvovv,r2)
        Hr1 +=  -einsum('aldc,lcd->a',imds.Wvovv,r2)

        # Eq. (31)
        Hr2 = einsum('abcj,c->jab',imds.Wvvvo,r1)
        Hr2 += einsum('ac,jcb->jab',imds.Lvv,r2)
        Hr2 += einsum('bd,jad->jab',imds.Lvv,r2)
        Hr2 -= einsum('lj,lab->jab',imds.Loo,r2)
        Hr2 += 2*einsum('lbdj,lad->jab',imds.Wovvo,r2)
        Hr2 +=  -einsum('lbjd,lad->jab',imds.Wovov,r2)
        Hr2 +=  -einsum('lajc,lcb->jab',imds.Wovov,r2)
        Hr2 +=  -einsum('lbcj,lca->jab',imds.Wovvo,r2)
        Hr2 += einsum('abcd,jcd->jab',imds.Wvvvv,r2)
        tmp = (2*einsum('klcd,lcd->k',imds.Woovv,r2)
                -einsum('kldc,lcd->k',imds.Woovv,r2))
        Hr2 -= einsum('k,kjab->jab',tmp,self.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = self.nea()
        vector = np.zeros(size, r1.dtype)
        vector[:nvir] = r1.copy()
        vector[nvir:] = r2.copy().reshape(nocc*nvir*nvir)
        return vector

    def eeccsd(self, nroots=1):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nee()
        nroots = min(nroots,size)
        self._eeconv, self.eee, evecs = eigs(self.eeccsd_matvec, size, nroots, verbose=log)
        if self._eeconv:
            logger.info(self, 'EE-CCSD converged')
        else:
            logger.info(self, 'EE-CCSD not converge')
        for n in range(nroots):
            logger.info(self, 'root %d E(EE-CCSD) = %.16g', n, self.eee.real[n])
        log.timer('EE-CCSD', *cput0)
        return self.eee.real[:nroots], evecs

    def eeccsd_matvec(self,vector):
        raise NotImplementedError

    def vector_to_amplitudes_ee(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nocc*nvir].copy().reshape((nocc,nvir))
        r2 = vector[nocc*nvir:].copy().reshape((nocc,nocc,nvir,nvir))
        return [r1,r2]

    def amplitudes_to_vector_ee(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = self.nee()
        vector = np.zeros(size, r1.dtype)
        vector[:nocc*nvir] = r1.copy().reshape(nocc*nvir)
        vector[nocc*nvir:] = r2.copy().reshape(nocc*nocc*nvir*nvir)
        return vector


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
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
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory)
            or cc.mol.incore_anyway):
            eri = ao2mofn(cc._scf.mol, (mo_coeff,mo_coeff,mo_coeff,mo_coeff), compact=0)
            if mo_coeff.dtype == np.float: eri = eri.real
            eri = eri.reshape((nmo,)*4)
            # <ij|kl> = (ik|jl)
            eri = eri.transpose(0,2,1,3)

            self.dtype = eri.dtype
            self.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
            self.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
            self.voov = eri[nocc:,:nocc,:nocc,nocc:].copy()
            self.vovv = eri[nocc:,:nocc,nocc:,nocc:].copy()
            self.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
        else:
            print "*** Using HDF5 ERI storage ***"
            _tmpfile1 = tempfile.NamedTemporaryFile()
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            if mo_coeff.dtype == np.complex: ds_type = 'c16'
            else: ds_type = 'f8'
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.voov = self.feri1.create_dataset('voov', (nvir,nocc,nocc,nvir), ds_type)
            self.vovv = self.feri1.create_dataset('vovv', (nvir,nocc,nvir,nvir), ds_type)
            self.vvvv = self.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)

            cput1 = time.clock(), time.time()
            # <ij|pq>  = (ip|jq)
            buf = ao2mofn(cc._scf.mol, (orbo,mo_coeff,orbo,mo_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nocc,nmo)).transpose(0,2,1,3)
            cput1 = log.timer_debug1('transforming oopq', *cput1)
            self.dtype = buf.dtype
            self.oooo[:,:,:,:] = buf[:,:,:nocc,:nocc]
            self.ooov[:,:,:,:] = buf[:,:,:nocc,nocc:]
            self.oovv[:,:,:,:] = buf[:,:,nocc:,nocc:]

            cput1 = time.clock(), time.time()
            # <ia|pq> = (ip|aq)
            buf = ao2mofn(cc._scf.mol, (orbo,mo_coeff,orbv,mo_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nvir,nmo)).transpose(0,2,1,3)
            cput1 = log.timer_debug1('transforming ovpq', *cput1)
            self.ovov[:,:,:,:] = buf[:,:,:nocc,nocc:]
            self.vovv[:,:,:,:] = buf[:,:,nocc:,nocc:].transpose(1,0,3,2)
            self.voov[:,:,:,:] = buf[:,:,nocc:,:nocc].transpose(1,0,3,2)

            for a in range(nvir):
                orbva = orbv[:,a].reshape(-1,1)
                buf = ao2mofn(cc._scf.mol, (orbva,orbv,orbv,orbv), compact=0)
                if mo_coeff.dtype == np.float: buf = buf.real
                buf = buf.reshape((1,nvir,nvir,nvir)).transpose(0,2,1,3)
                self.vvvv[a] = buf[:]

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)

class _IMDS:
    def __init__(self, cc):
        self.cc = cc
        self._made_shared = False

    def _make_shared(self):
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
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
        raise NotImplementedError


def _mem_usage(nocc, nvir):
    incore = (nocc**4 + 2*nocc**3*nvir
        + 2*nocc**2*nvir**2 + nocc*nvir**3 + nvir**4)*8/1e6
    # Additional ERIs to be removed, by symmetry
    incore += (nocc**3*nvir + nocc**2*nvir**2 + nocc*nvir**3)*8/1e6
    # Roughly, factor of two for intermediates and factor of two 
    # for safety (temp arrays, copying, etc)
    incore *= 4
    outcore = basic = incore
    return incore, outcore, basic
