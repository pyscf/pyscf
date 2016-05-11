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
from pyscf.cc.ccsd import _cp
from pyscf.cc import rintermediates as imd
from pyscf.pbc.lib.linalg_helper import eigs

#einsum = np.einsum
einsum = pbclib.einsum

# This is restricted (R)CCSD
# following Hirata, ..., Barlett, J. Chem. Phys. 120, 2581 (2004)

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
    t1new +=  einsum('ac,ic->ia',Fvv,t1)
    t1new += -einsum('ki,ka->ia',Foo,t1)
    t1new +=  einsum('kc,kica->ia',Fov,2*t2)
    t1new +=  einsum('kc,ikca->ia',Fov, -t2)
    t1new +=  einsum('kc,ic,ka->ia',Fov,t1,t1)
    t1new +=  einsum('akic,kc->ia',2*eris.voov,t1)
    t1new +=  einsum('akci,kc->ia', -eris.vovo,t1)
    t1new +=  einsum('akcd,ikcd->ia',2*eris.vovv,t2)
    t1new +=  einsum('akdc,ikcd->ia', -eris.vovv,t2)
    t1new +=  einsum('akcd,ic,kd->ia',2*eris.vovv,t1,t1)
    t1new +=  einsum('akdc,ic,kd->ia', -eris.vovv,t1,t1)
    t1new += -einsum('klic,klac->ia',2*eris.ooov,t2)
    t1new += -einsum('klci,klac->ia', -eris.oovo,t2)
    t1new += -einsum('klic,ka,lc->ia',2*eris.ooov,t1,t1)
    t1new += -einsum('klci,ka,lc->ia', -eris.oovo,t1,t1)

    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36)
    t2new = np.array(eris.oovv, copy=True).conj()
    t2new += einsum('klij,klab->ijab',Woooo,t2)
    t2new += einsum('klij,ka,lb->ijab',Woooo,t1,t1)
    t2new += einsum('abcd,ijcd->ijab',Wvvvv,t2)
    t2new += einsum('abcd,ic,jd->ijab',Wvvvv,t1,t1)
    tmp = einsum('ac,ijcb->ijab',Lvv,t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('ki,kjab->ijab',Loo,t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp2 = eris.vvov - einsum('kbic,ka->abic',eris.ovov,t1)
    tmp = einsum('abic,jc->ijab',tmp2,t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2 = eris.vooo + einsum('akic,jc->akij',eris.voov,t1)
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
        self.made_ee_imds = False
        self.made_ip_imds = False
        self.made_ea_imds = False

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** EOM CC flags ********')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc()
        #t1 = np.zeros((nocc,nvir), eris.dtype)
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        eris_oovv = np.array(eris.oovv, copy=True)
        t1 = eris.fock[:nocc,nocc:] / eia
        woovv = 2*eris_oovv - eris_oovv.transpose(0,1,3,2)
        t2 = eris_oovv/eijab
        self.emp2 = einsum('ijab,ijab',t2,woovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
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

    def eeccsd(self, nroots=2*4):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nocc*nvir+nocc*(nocc-1)/2*nvir*(nvir-1)/2
        evals, evecs = eigs(self.eeccsd_matvec, size, nroots)
        return evals.real[:nroots], evecs

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

        if not self.made_ee_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ee()
            self.made_ee_imds = True

        imds = self.imds

        # Eq. (9)
        Hr1 = ( einsum('ae,ie->ia',imds.Fvv,r1)
               - einsum('mi,ma->ia',imds.Foo,r1)
               + einsum('me,imae->ia',imds.Fov,r2)
               + einsum('amie,me->ia',imds.Wvoov,r1)
               - 0.5*einsum('mnie,mnae->ia',imds.Wooov,r2)
               + 0.5*einsum('amef,imef->ia',imds.Wvovv,r2) )
        # Eq. (10)
        tmpab = ( einsum('be,ijae->ijab',imds.Fvv,r2)
                 - 0.5*einsum('mnef,ijae,mnbf->ijab',imds.Woovv,self.t2,r2)
                 - einsum('mbij,ma->ijab',imds.Wovoo,r1)
                 + einsum('maef,ijfb,me->ijab',imds.Wovvv,self.t2,r1) )
        tmpij = (-einsum('mj,imab->ijab',imds.Foo,r2)
                 - 0.5*einsum('mnef,imab,jnef->ijab',imds.Woovv,self.t2,r2)
                 + einsum('abej,ie->ijab',imds.Wvvvo,r1)
                 - einsum('mnei,njab,me->ijab',imds.Woovo,self.t2,r1) )

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

    def ipccsd(self, nroots=2*4):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nocc + nocc*nocc*nvir
        evals, evecs = eigs(self.ipccsd_matvec, size, nroots)
        return evals.real[:nroots], evecs

    def ipccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # Z. Tu, F. Wang, and X. Li                            #
    # J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ip(vector)

        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip()
            self.made_ip_imds = True

        imds = self.imds

        Hr1 = -einsum('ki,k->i',imds.Loo,r1)
        Hr1 += 2.*einsum('ld,ild->i',imds.Fov,r2)
        Hr1 +=   -einsum('kd,kid->i',imds.Fov,r2)
        Hr1 += -2.*einsum('klid,kld->i',imds.Wooov,r2)
        Hr1 +=     einsum('lkid,kld->i',imds.Wooov,r2)

        tmp = 2.*einsum('lkdc,kld->c',imds.Woovv,r2)
        tmp +=  -einsum('kldc,kld->c',imds.Woovv,r2)
        Hr2 = einsum('bd,ijd->ijb',imds.Lvv,r2)
        Hr2 += -einsum('ki,kjb->ijb',imds.Loo,r2)
        Hr2 += -einsum('lj,ilb->ijb',imds.Loo,r2)
        Hr2 += -einsum('kbij,k->ijb',imds.Wovoo,r1)
        Hr2 +=  einsum('klij,klb->ijb',imds.Woooo,r2)
        Hr2 += 2.*einsum('lbdj,ild->ijb',imds.Wovvo,r2)
        Hr2 +=   -einsum('kbdj,kid->ijb',imds.Wovvo,r2)
        Hr2 +=   -einsum('lbjd,ild->ijb',imds.Wovov,r2) #typo in nooijen's paper
        Hr2 +=   -einsum('kbid,kjd->ijb',imds.Wovov,r2)
        Hr2 += -einsum('c,ijcb->ijb',tmp,self.t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
        #r2 = np.zeros((nocc,nocc,nvir), vector.dtype)
        #index = nocc
        #for i in range(nocc):
        #    for j in range(nocc):
        #        for a in range(nvir):
        #            r2[i,j,a] =  vector[index]
        #            index += 1
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nocc+nocc*nocc*nvir
        vector = np.zeros(size, r1.dtype)
        vector[:nocc] = r1.copy()
        vector[nocc:] = r2.copy().reshape(nocc*nocc*nvir)
        #index = nocc
        #for i in range(nocc):
        #    for j in range(nocc):
        #        for a in range(nvir):
        #            vector[index] = r2[i,j,a]
        #            index += 1
        return vector

    def eaccsd(self, nroots=2*4):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nvir + nvir*nvir*nocc
        evals, evecs = eigs(self.eaccsd_matvec, size, nroots)
        return evals.real[:nroots], evecs

    def eaccsd_matvec(self,vector):
    ########################################################
    # FOLLOWING:                                           #
    # M. Nooijen and R. J. Bartlett,                       #
    # J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ea(vector)

        if not self.made_ea_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ea()
            self.made_ea_imds = True

        imds = self.imds

        # Eq. (30)
        Hr1 =  einsum('ac,c->a',imds.Lvv,r1)
        Hr1 += einsum('ld,lad->a',2.*imds.Fov,r2)
        Hr1 += einsum('ld,lda->a',  -imds.Fov,r2)
        Hr1 += 2.*einsum('alcd,lcd->a',imds.Wvovv,r2)
        Hr1 +=   -einsum('aldc,lcd->a',imds.Wvovv,r2)

        ## Eq. (31)
        Hr2 = einsum('abcj,c->jab',imds.Wvvvo,r1)
        Hr2 += einsum('ac,jcb->jab',imds.Lvv,r2)
        Hr2 += einsum('bd,jad->jab',imds.Lvv,r2)
        Hr2 -= einsum('lj,lab->jab',imds.Loo,r2)
        Hr2 += 2.*einsum('lbdj,lad->jab',imds.Wovvo,r2)
        Hr2 +=   -einsum('bldj,lad->jab',imds.Wvovo,r2)
        Hr2 +=   -einsum('alcj,lcb->jab',imds.Wvovo,r2)
        Hr2 +=   -einsum('bljc,lca->jab',imds.Wovvo.transpose(1,0,3,2),r2)
        Hr2 += einsum('abcd,jcd->jab',imds.Wvvvv,r2)
        tmp = (2.*einsum('klcd,lcd->k',imds.Woovv,r2)
                 -einsum('kldc,lcd->k',imds.Woovv,r2))
        Hr2 -= einsum('k,kjab->jab',tmp,self.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
        #r2 = np.zeros((nocc,nvir,nvir), vector.dtype)
        #index = nvir
        #for i in range(nocc):
        #    for a in range(nvir):
        #        for b in range(nvir):
        #            r2[i,a,b] =  vector[index]
        #            index += 1
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        size = nvir+nvir*nvir*nocc
        vector = np.zeros(size, r1.dtype)
        vector[:nvir] = r1.copy()
        vector[nvir:] = r2.copy().reshape(nocc*nvir*nvir)
        #index = nvir
        #for i in range(nocc):
        #    for a in range(nvir):
        #        for b in range(nvir):
        #            vector[index] = r2[i,a,b]
        #            index += 1
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
        mem_incore, mem_outcore, mem_basic = pyscf.cc.ccsd._mem_usage(nocc, nvir)
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
            #self.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
            self.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
            self.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()

            # TODO: Avoid this.
            # Store all for now, while DEBUGGING
            self.voov = eri[nocc:,:nocc,:nocc,nocc:].copy()
            self.vovo = eri[nocc:,:nocc,nocc:,:nocc].copy()
            self.vovv = eri[nocc:,:nocc,nocc:,nocc:].copy()
            self.oovo = eri[:nocc,:nocc,nocc:,:nocc].copy()
            self.vvov = eri[nocc:,nocc:,:nocc,nocc:].copy()
            self.vooo = eri[nocc:,:nocc,:nocc,:nocc].copy()

            #ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri[nocc:,nocc:,nocc:,nocc:], nvir)

        log.timer('CCSD integral transformation', *cput0)

class _IMDS:
    def __init__(self, cc):
        self.cc = cc

    def make_ee(self):
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

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
        self.Woovv = eris.oovv
        self.Wvoov =  self.imds.Wovvo.transpose(1,0,3,2)
        self.Wovvv = -self.imds.Wvovv.transpose(1,0,2,3)
        self.Woovo = -self.imds.Wooov.transpose(0,1,3,2)

    def make_ip(self):
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Loo = imd.Loo(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Woovv = eris.oovv

    def make_ea(self):
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Loo = imd.Loo(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Wvvvv = imd.Wvvvv(t1,t2,eris)
        self.Woovv = eris.oovv
        self.Wvovo = imd.Wovov(t1,t2,eris).transpose(1,0,3,2)
