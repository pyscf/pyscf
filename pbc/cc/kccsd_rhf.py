import time
import numpy
import numpy as np
import kpoint_helper

import pyscf.pbc.tools.pbc as tools
from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.pbc.cc import kintermediates_rhf as imdk
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
    nkpts, nocc, nvir = t1.shape
    eold = 0.0
    eccsd = 0.0
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


#@profile
def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:,:nocc,nocc:]
    foo = fock[:,:nocc,:nocc]
    fvv = fock[:,nocc:,nocc:]

    #mo_e = eris.fock.diagonal()
    #eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    #eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    Foo = imdk.cc_Foo(cc,t1,t2,eris)
    Fvv = imdk.cc_Fvv(cc,t1,t2,eris)
    Fov = imdk.cc_Fov(cc,t1,t2,eris)
    Loo = imdk.Loo(cc,t1,t2,eris)
    Lvv = imdk.Lvv(cc,t1,t2,eris)
    Woooo = imdk.cc_Woooo(cc,t1,t2,eris)
    Wvvvv = imdk.cc_Wvvvv(cc,t1,t2,eris)
    Wvoov = imdk.cc_Wvoov(cc,t1,t2,eris)
    Wvovo = imdk.cc_Wvovo(cc,t1,t2,eris)

    # Move energy terms to the other side
    Foo -= foo
    Fvv -= fvv
    Loo -= foo
    Lvv -= fvv

    kconserv = cc.kconserv

    # T1 equation
    # TODO: Check this conj(). Hirata and Bartlett has
    # f_{vo}(a,i), which should be equal to f_{ov}^*(i,a)
    t1new = np.empty((nkpts,nocc,nvir),dtype=t1.dtype)
    t1new[:] = fov[:].conj().copy()
    for ka in range(nkpts):
        ki = ka
        # kc == ki; kk == ka
        t1new[ka] += -2.*einsum('kc,ka,ic->ia',fov[ki],t1[ka],t1[ki])
        t1new[ka] += einsum('ac,ic->ia',Fvv[ka],t1[ki])
        t1new[ka] += -einsum('ki,ka->ia',Foo[ki],t1[ka])

        tau_term = np.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t1.dtype)
        for kk in range(nkpts):
            tau_term[kk] = 2*t2[kk,ki,kk] - t2[ki,kk,kk].transpose(1,0,2,3)
        tau_term[ka] += einsum('ic,ka->kica',t1[ki],t1[ka])

        for kk in range(nkpts):
            kc = kk
            t1new[ka] += einsum('kc,kica->ia',Fov[kc],tau_term[kk])

            t1new[ka] +=  einsum('akic,kc->ia',2*eris.voov[ka,kk,ki],t1[kc])
            t1new[ka] +=  einsum('akci,kc->ia', -eris.vovo[ka,kk,kc],t1[kc])

            for kc in range(nkpts):
                kd = kconserv[ka,kc,kk]

                Svovv = 2*eris.vovv[ka,kk,kc] - eris.vovv[ka,kk,kd].transpose(0,1,3,2)
                tau_term_1 = t2[ki,kk,kc].copy()
                if ki == kc and kk == kd:
                    tau_term_1 += einsum('ic,kd->ikcd',t1[ki],t1[kk])
                t1new[ka] += einsum('akcd,ikcd->ia',Svovv,tau_term_1)

                # kk - ki + kl = kc
                #  => kl = ki - kk + kc
                kl = kconserv[ki,kk,kc]
                Sooov = 2*eris.ooov[kk,kl,ki] - eris.ooov[kl,kk,ki].transpose(1,0,2,3)
                tau_term_1 = t2[kk,kl,ka].copy()
                if kk == ka and kl == kc:
                    tau_term_1 += einsum('ka,lc->klac',t1[ka],t1[kc])
                t1new[ka] += -einsum('klic,klac->ia',Sooov,tau_term_1)

    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36)
    t2new = np.array(eris.oovv, copy=True).conj()
    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
            kb = kconserv[ki,ka,kj]

            for kl in range(nkpts):
                # kk - ki + kl = kj
                # => kk = kj - kl + ki
                kk = kconserv[kj,kl,ki]
                t2new[ki,kj,ka] += einsum('klij,klab->ijab',Woooo[kk,kl,ki],t2[kk,kl,ka])
                if kl == kb and kk == ka:
                    t2new[ki,kj,ka] += einsum('klij,ka,lb->ijab',Woooo[ka,kb,ki],t1[ka],t1[kb])

            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                tau_term = t2[ki,kj,kc].copy()
                if ki == kc and kj == kd:
                    tau_term += einsum('ic,jd->ijcd',t1[ki],t1[kj])
                t2new[ki,kj,ka] += einsum('abcd,ijcd->ijab',Wvvvv[ka,kb,kc],tau_term)

            t2new[ki,kj,ka] += einsum('ac,ijcb->ijab',Lvv[ka],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('bc,jica->ijab',Lvv[kb],t2[kj,ki,kb])

            t2new[ki,kj,ka] += einsum('ki,kjab->ijab',-Loo[ki],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('kj,kiba->ijab',-Loo[kj],t2[kj,ki,kb])

            tmp2 = eris.vvov[ka,kb,ki] - einsum('kbic,ka->abic',eris.ovov[ka,kb,ki],t1[ka])
            tmp  = einsum('abic,jc->ijab',tmp2,t1[kj])
            t2new[ki,kj,ka] += tmp
            #P(ij)P(ab)
            tmp2 = eris.vvov[kb,ka,kj] - einsum('kajc,kb->bajc',eris.ovov[kb,ka,kj],t1[kb])
            tmp  = einsum('bajc,ic->ijab',tmp2,t1[ki])
            t2new[ki,kj,ka] += tmp

            # ka - ki + kk = kj
            # => kk = ki - ka + kj
            kk = kconserv[ki,ka,kj]
            tmp2 = eris.vooo[ka,kk,ki] + einsum('akic,jc->akij',eris.voov[ka,kk,ki],t1[kj])
            tmp  = einsum('akij,kb->ijab',tmp2,t1[kb])
            t2new[ki,kj,ka] -= tmp
            #P(ij)P(ab)
            kk = kconserv[kj,kb,ki]
            tmp2 = eris.vooo[kb,kk,kj] + einsum('bkjc,ic->bkji',eris.voov[kb,kk,kj],t1[ki])
            tmp  = einsum('bkji,ka->ijab',tmp2,t1[ka])
            t2new[ki,kj,ka] -= tmp

            for kk in range(nkpts):
                kc = kconserv[ka,ki,kk]
                tmp_voov = 2.*Wvoov[ka,kk,ki] - Wvovo[ka,kk,kc].transpose(0,1,3,2)
                tmp = einsum('akic,kjcb->ijab',tmp_voov,t2[kk,kj,kc])
                #tmp = 2*einsum('akic,kjcb->ijab',Wvoov[ka,kk,ki],t2[kk,kj,kc]) - \
                #        einsum('akci,kjcb->ijab',Wvovo[ka,kk,kc],t2[kk,kj,kc])
                t2new[ki,kj,ka] += tmp
                #P(ij)P(ab)
                kc = kconserv[kb,kj,kk]
                tmp_voov = 2.*Wvoov[kb,kk,kj] - Wvovo[kb,kk,kc].transpose(0,1,3,2)
                tmp = einsum('bkjc,kica->ijab',tmp_voov,t2[kk,ki,kc])
                #tmp = 2*einsum('bkjc,kica->ijab',Wvoov[kb,kk,kj],t2[kk,ki,kc]) - \
                #        einsum('bkcj,kica->ijab',Wvovo[kb,kk,kc],t2[kk,ki,kc])
                t2new[ki,kj,ka] += tmp

                kc = kconserv[ka,ki,kk]
                tmp = einsum('akic,kjbc->ijab',Wvoov[ka,kk,ki],t2[kk,kj,kb])
                t2new[ki,kj,ka] -= tmp
                #P(ij)P(ab)
                kc = kconserv[kb,kj,kk]
                tmp = einsum('bkjc,kiac->ijab',Wvoov[kb,kk,kj],t2[kk,ki,ka])
                t2new[ki,kj,ka] -= tmp

                kc = kconserv[kk,ka,kj]
                tmp = einsum('bkci,kjac->ijab',Wvovo[kb,kk,kc],t2[kk,kj,ka])
                t2new[ki,kj,ka] -= tmp
                #P(ij)P(ab)
                kc = kconserv[kk,kb,ki]
                tmp = einsum('akcj,kibc->ijab',Wvovo[ka,kk,kc],t2[kk,ki,kb])
                t2new[ki,kj,ka] -= tmp

    eia = numpy.zeros(shape=t1new.shape, dtype=t1new.dtype)
    for ki in range(nkpts):
        for i in range(nocc):
            for a in range(nvir):
                eia[ki,i,a] = foo[ki,i,i] - fvv[ki,a,a]
        t1new[ki] /= eia[ki]

    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[ki,ka,kj]
            eia = np.diagonal(foo[ki]).reshape(-1,1) - np.diagonal(fvv[ka])
            ejb = np.diagonal(foo[kj]).reshape(-1,1) - np.diagonal(fvv[kb])
            eijab = pyscf.lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2new[ki,kj,ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)
#    sys.exit("exiting for testing...")

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    fock = eris.fock
    e = 0.0 + 1j*0.0
    for ki in range(nkpts):
        e += 2*einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki])
    t1t1 = numpy.zeros(shape=t2.shape,dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki,kj,ka] = einsum('ia,jb->ijab',t1[ki],t1[kj])
    tau = t2 + t1t1
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e += einsum('ijab,ijab', 2*tau[ki,kj,ka], eris.oovv[ki,kj,ka])
                e += einsum('ijab,ijba',  -tau[ki,kj,ka], eris.oovv[ki,kj,kb])
    e /= nkpts
    return e.real


class RCCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, abs_kpts, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)
        self._kpts = abs_kpts
        self.nkpts = len(self._kpts)
        self.kconserv = tools.get_kconserv(mf.cell, abs_kpts)
        self.khelper = kpoint_helper.unique_pqr_list(mf.cell, abs_kpts)
        self.made_ee_imds = False
        self.made_ip_imds = False
        self.made_ea_imds = False

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** EOM CC flags ********')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = len(self._kpts)
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=numpy.complex128)
        t2 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        woovv = numpy.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        self.emp2 = 0
        foo = eris.fock[:,:nocc,:nocc].copy()
        fvv = eris.fock[:,nocc:,nocc:].copy()
        eris_oovv = eris.oovv.copy()
        eia = numpy.zeros((nocc,nvir))
        eijab = numpy.zeros((nocc,nocc,nvir,nvir))

        kconserv = self.kconserv
        for ki in range(nkpts):
          for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                eia = np.diagonal(foo[ki]).reshape(-1,1) - np.diagonal(fvv[ka])
                ejb = np.diagonal(foo[kj]).reshape(-1,1) - np.diagonal(fvv[kb])
                eijab = pyscf.lib.direct_sum('ia,jb->ijab',eia,ejb)
                woovv[ki,kj,ka] = (2*eris_oovv[ki,kj,ka] - eris_oovv[ki,kj,kb].transpose(0,1,3,2))
                t2[ki,kj,ka] = eris_oovv[ki,kj,ka] / eijab

        t2 = numpy.conj(t2)
        self.emp2 = numpy.einsum('pqrijab,pqrijab',t2,woovv).real
        self.emp2 /= nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def nocc(self):
        # Spin orbitals
        # TODO: Possibly change this to make it work with k-points with frozen
        #       As of right now it works, but just not sure how the frozen list will work
        #       with it
        self._nocc = pyscf.cc.ccsd.CCSD.nocc(self)
        self._nocc = (self._nocc // len(self._kpts))
        return self._nocc

    def nmo(self):
        # TODO: Change this for frozen at k-points, seems like it should work
        if isinstance(self.frozen, (int, numpy.integer)):
            self._nmo = len(self.mo_energy[0]) - self.frozen
        else:
            if len(self.frozen) > 0:
                self._nmo = len(self.mo_energy[0]) - len(self.frozen[0])
            else:
                self._nmo = len(self.mo_energy[0])
        return self._nmo

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

    def ipccsd(self, nroots=2*4):
        time0 = time.clock(), time.time()
        log = logger.Logger(self.stdout, self.verbose)
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size =  nocc + nkpts*nkpts*nocc*nocc*nvir
        evals = np.zeros((nkpts,nroots),np.complex)
        evecs = np.zeros((nkpts,size,nroots),np.complex)
        for kshift in range(nkpts):
            self.kshift = kshift
            evals[kshift], evecs[kshift] = eigs(self.ipccsd_matvec, size, nroots, self.ipccsd_diag())
            #np.set_printoptions(precision=16)
            #print "kshift evals : ", evals[kshift]
        time0 = log.timer_debug1('converge ip-ccsd', *time0)
        return evals.real, evecs

    #@profile
    def ipccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # Z. Tu, F. Wang, and X. Li                            #
    # J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ip(vector)

        t1,t2 = self.t1, self.t2
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip()
            self.made_ip_imds = True

        imds = self.imds

        Hr1 = -einsum('ki,k->i',imds.Loo[kshift],r1)
        for kl in range(nkpts):
            Hr1 += 2.*einsum('ld,ild->i',imds.Fov[kl],r2[kshift,kl])
            Hr1 +=   -einsum('ld,lid->i',imds.Fov[kl],r2[kl,kshift])
            for kk in range(nkpts):
                kd = kconserv[kk,kshift,kl]
                Hr1 += -2.*einsum('klid,kld->i',imds.Wooov[kk,kl,kshift],r2[kk,kl])
                Hr1 +=     einsum('lkid,kld->i',imds.Wooov[kl,kk,kshift],r2[kk,kl])

        Hr2 = np.zeros(r2.shape,dtype=t1.dtype)
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                Hr2[ki,kj] += einsum('bd,ijd->ijb',imds.Lvv[kb],r2[ki,kj])
                Hr2[ki,kj] -= einsum('li,ljb->ijb',imds.Loo[ki],r2[ki,kj])
                Hr2[ki,kj] -= einsum('lj,ilb->ijb',imds.Loo[kj],r2[ki,kj])
                Hr2[ki,kj] -= einsum('kbij,k->ijb',imds.Wovoo[kshift,kb,ki],r1)
                for kl in range(nkpts):
                    kk = kconserv[ki,kl,kj]
                    Hr2[ki,kj] += einsum('klij,klb->ijb',imds.Woooo[kk,kl,ki],r2[kk,kl])
                    kd = kconserv[kl,kj,kb]
                    Hr2[ki,kj] += 2.*einsum('lbdj,ild->ijb',imds.Wovvo[kl,kb,kd],r2[ki,kl])
                    Hr2[ki,kj] += -einsum('lbdj,lid->ijb',imds.Wovvo[kl,kb,kd],r2[kl,ki])
                    Hr2[ki,kj] += -einsum('lbjd,ild->ijb',imds.Wovov[kl,kb,kj],r2[ki,kl]) #typo in nooijen's paper
                    kd = kconserv[kl,ki,kb]
                    Hr2[ki,kj] += -einsum('lbid,ljd->ijb',imds.Wovov[kl,kb,ki],r2[kl,kj])
                    for kk in range(nkpts):
                        kc = kshift
                        kd = kconserv[kl,kc,kk]
                        tmp = ( 2.*einsum('lkdc,kld->c',imds.Woovv[kl,kk,kd],r2[kk,kl])
                                  -einsum('kldc,kld->c',imds.Woovv[kk,kl,kd],r2[kk,kl]) )
                        Hr2[ki,kj] += -einsum('c,ijcb->ijb',tmp,t2[ki,kj,kshift])

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def ipccsd_diag(self):
        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip()
            self.made_ip_imds = True

        imds = self.imds

        Hr1 = -np.diag(imds.Loo[kshift])

        Hr2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=t1.dtype)
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                for i in range(nocc):
                    for j in range(nocc):
                        for b in range(nvir):
                            Hr2[ki,kj,i,j,b] = imds.Lvv[kb,b,b]
                            Hr2[ki,kj,i,j,b] -= imds.Loo[ki,i,i]
                            Hr2[ki,kj,i,j,b] -= imds.Loo[kj,j,j]
                            for kl in range(nkpts):
                                kk = kconserv[ki,kl,kj]
                                Hr2[ki,kj,i,j,b] += imds.Woooo[kk,kl,ki,i,j,i,j]*(kk==ki)*(kl==kj)
                                kd = kconserv[kl,kj,kb]
                                Hr2[ki,kj,i,j,b] += 2.*imds.Wovvo[kl,kb,kd,j,b,b,j]*(kl==kj)
                                Hr2[ki,kj,i,j,b] += -imds.Wovvo[kl,kb,kd,i,b,b,j]*(i==j)*(kl==ki)*(ki==kj)
                                Hr2[ki,kj,i,j,b] += -imds.Wovov[kl,kb,kj,j,b,j,b]*(kl==kj)
                                kd = kconserv[kl,ki,kb]
                                Hr2[ki,kj,i,j,b] += -imds.Wovov[kl,kb,ki,i,b,i,b]*(kl==ki)
                                for kk in range(nkpts):
                                    kc = kshift
                                    kd = kconserv[kl,kc,kk]
                                    Hr2[ki,kj,i,j,b] += -2.*np.dot(t2[ki,kj,kshift,i,j,:,b],imds.Woovv[kl,kk,kd,j,i,b,:])*(kk==ki)*(kl==kj)
                                    Hr2[ki,kj,i,j,b] += np.dot(t2[ki,kj,kshift,i,j,:,b],imds.Woovv[kk,kl,kd,i,j,b,:])*(kk==ki)*(kl==kj)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts

        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
        #r2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir), vector.dtype)
        #index = nocc
        #for ki in range(nkpts):
        #    for kj in range(nkpts):
        #        for i in range(nocc):
        #            for j in range(nocc):
        #                for a in range(nvir):
        #                    r2[ki,kj,i,j,a] =  vector[index]
        #                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size = nocc + nkpts*nkpts*nocc*nocc*nvir

        vector = np.zeros((size), r1.dtype)
        vector[:nocc] = r1.copy()
        vector[nocc:] = r2.copy().reshape(nkpts*nkpts*nocc*nocc*nvir)
        #index = nocc
        #for ki in range(nkpts):
        #    for kj in range(nkpts):
        #        for i in range(nocc):
        #            for j in range(nocc):
        #                for a in range(nvir):
        #                    vector[index] = r2[ki,kj,i,j,a]
        #                    index += 1
        return vector

    def eaccsd(self, nroots=2*4):
        time0 = time.clock(), time.time()
        log = logger.Logger(self.stdout, self.verbose)
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size =  nvir + nkpts*nkpts*nocc*nvir*nvir
        evals = np.zeros((nkpts,nroots),np.complex)
        evecs = np.zeros((nkpts,size,nroots),np.complex)
        for kshift in range(nkpts):
            self.kshift = kshift
            evals[kshift], evecs[kshift] = eigs(self.eaccsd_matvec, size, nroots, self.eaccsd_diag())
            #np.set_printoptions(precision=16)
            #print "kshift evals : ", evals[:nroots]
        time0 = log.timer_debug1('converge ea-ccsd', *time0)
        return evals.real, evecs

    #@profile
    def eaccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # M. Nooijen and R. J. Bartlett,                       #
    # J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ea(vector)

        t1,t2 = self.t1, self.t2
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ea_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ea()
            self.made_ea_imds = True

        imds = self.imds

        Hr1 = einsum('ac,c->a',imds.Lvv[kshift],r1)
        for kl in range(nkpts):
            Hr1 += 2.*einsum('ld,lad->a',imds.Fov[kl],r2[kl,kshift])
            Hr1 +=   -einsum('ld,lda->a',imds.Fov[kl],r2[kl,kl])
            for kc in range(nkpts):
                kd = kconserv[kshift,kc,kl]
                Hr1 +=  2.*einsum('alcd,lcd->a',imds.Wvovv[kshift,kl,kc],r2[kl,kc])
                Hr1 +=    -einsum('aldc,lcd->a',imds.Wvovv[kshift,kl,kd],r2[kl,kc])

        Hr2 = np.zeros(r2.shape,dtype=t1.dtype)
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                Hr2[kj,ka] += einsum('abcj,c->jab',imds.Wvvvo[ka,kb,kshift],r1)
                Hr2[kj,ka] -= einsum('lj,lab->jab',imds.Loo[kj],r2[kj,ka])
                Hr2[kj,ka] += einsum('ac,jcb->jab',imds.Lvv[ka],r2[kj,ka])
                Hr2[kj,ka] += einsum('bd,jad->jab',imds.Lvv[kb],r2[kj,ka])
                for kd in range(nkpts):
                    kc = kconserv[ka,kd,kb]
                    Hr2[kj,ka] += einsum('abcd,jcd->jab',imds.Wvvvv[ka,kb,kc],r2[kj,kc])
                    kl = kconserv[kd,kb,kj]
                    Hr2[kj,ka] += 2.*einsum('lbdj,lad->jab',imds.Wovvo[kl,kb,kd],r2[kl,ka])
                    #imds.Wvovo[kb,kl,kd,kj] <= imds.Wovov[kl,kb,kj,kd].transpose(1,0,3,2)
                    Hr2[kj,ka] += -einsum('bldj,lad->jab',imds.Wovov[kl,kb,kj].transpose(1,0,3,2),r2[kl,ka])
                    #imds.Wvoov[kb,kl,kj,kd] <= imds.Wovvo[kl,kb,kd,kj].transpose(1,0,3,2)
                    Hr2[kj,ka] += -einsum('bljd,lda->jab',imds.Wovvo[kl,kb,kd].transpose(1,0,3,2),r2[kl,kd])
                    kl = kconserv[kd,ka,kj]
                    #imds.Wvovo[ka,kl,kd,kj] <= imds.Wovov[kl,ka,kj,kd].transpose(1,0,3,2)
                    Hr2[kj,ka] += -einsum('aldj,ldb->jab',imds.Wovov[kl,ka,kj].transpose(1,0,3,2),r2[kl,kd])
                    for kc in range(nkpts):
                        kk = kshift
                        kl = kconserv[kc,kk,kd]
                        tmp = ( 2.*einsum('klcd,lcd->k',imds.Woovv[kk,kl,kc],r2[kl,kc])
                                  -einsum('kldc,lcd->k',imds.Woovv[kk,kl,kd],r2[kl,kc]) )
                        Hr2[kj,ka] += -einsum('k,kjab->jab',tmp,t2[kshift,kj,ka])

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd_diag(self):
        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ea_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ea()
            self.made_ea_imds = True

        imds = self.imds

        Hr1 = np.diag(imds.Lvv[kshift])

        Hr2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=t1.dtype)
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            Hr2[kj,ka,j,a,b] -= imds.Loo[kj,j,j]
                            Hr2[kj,ka,j,a,b] += imds.Lvv[ka,a,a]
                            Hr2[kj,ka,j,a,b] += imds.Lvv[kb,b,b]
                            for kd in range(nkpts):
                                kc = kconserv[ka,kd,kb]
                                Hr2[kj,ka,j,a,b] += imds.Wvvvv[ka,kb,kc,a,b,a,b]*(kc==ka)
                                kl = kconserv[kd,kb,kj]
                                Hr2[kj,ka,j,a,b] += 2.*imds.Wovvo[kl,kb,kd,j,b,b,j]*(kl==kj)
                                Hr2[kj,ka,j,a,b] += -imds.Wovov[kl,kb,kj].transpose(1,0,3,2)[b,j,b,j]*(kl==kj)
                                Hr2[kj,ka,j,a,b] += -imds.Wovvo[kl,kb,kd].transpose(1,0,3,2)[b,j,j,b]*(a==b)*(kl==kj)*(kd==ka)
                                kl = kconserv[kd,ka,kj]
                                Hr2[kj,ka,j,a,b] += -imds.Wovov[kl,ka,kj].transpose(1,0,3,2)[a,j,a,j]*(kl==kj)*(kd==ka)
                                for kc in range(nkpts):
                                    kk = kshift
                                    kl = kconserv[kc,kk,kd]
                                    Hr2[kj,ka,j,a,b] += -2*np.dot(t2[kshift,kj,ka,:,j,a,b],imds.Woovv[kk,kl,kc,:,j,a,b])*(kl==kj)*(kc==ka)
                                    Hr2[kj,ka,j,a,b] += np.dot(t2[kshift,kj,ka,:,j,a,b],imds.Woovv[kk,kl,kd,:,j,b,a])*(kl==kj)*(kc==ka)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts

        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
        #r2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir), vector.dtype)
        #index = nvir
        #for kj in range(nkpts):
        #    for ka in range(nkpts):
        #        for j in range(nocc):
        #            for a in range(nvir):
        #                for b in range(nvir):
        #                    r2[kj,ka,j,a,b] = vector[index]
        #                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size = nvir + nkpts*nkpts*nocc*nvir*nvir

        vector = np.zeros((size), r1.dtype)
        vector[:nvir] = r1.copy()
        vector[nvir:] = r2.copy().reshape(nkpts*nkpts*nocc*nvir*nvir)
        #index = nvir
        #for kj in range(nkpts):
        #    for ka in range(nkpts):
        #        for j in range(nocc):
        #            for a in range(nvir):
        #                for b in range(nvir):
        #                    vector[index] = r2[kj,ka,j,a,b]
        #                    index += 1
        return vector

class _ERIS:
    #@profile
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(cc.mo_energy.shape, dtype=numpy.bool)
        nkpts = len(cc._kpts)
        nmo = cc.nmo()
        #TODO check that this and kccsd work for frozen...
        if isinstance(cc.frozen, (int, numpy.integer)):
            moidx[:,:cc.frozen] = False
        elif len(cc.frozen) > 0:
            moidx[:,numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = numpy.zeros((nkpts,nmo,nmo),dtype=cc.mo_coeff.dtype)
            for kp in range(nkpts):
                self.mo_coeff[kp] = cc.mo_coeff[kp][:,moidx[kp]]
            mo_coeff = self.mo_coeff
            self.fock = numpy.zeros((nkpts,nmo,nmo),dtype=cc.mo_coeff.dtype)
            for kp in range(nkpts):
                self.fock[kp] = numpy.diag(cc.mo_energy[kp][moidx[kp]]).astype(mo_coeff.dtype)
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,:,moidx]
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
            kconserv = cc.kconserv
            khelper = cc.khelper #kpoint_helper.unique_pqr_list(cc._scf.cell,cc._kpts)
            unique_klist = khelper.get_uniqueList()
            nUnique_klist = khelper.nUnique

            eri = numpy.zeros((nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo), dtype=numpy.complex128)

            #
            #
            # Looping over unique list of k-vectors
            #
            #
            for pqr in range(nUnique_klist):
                kp, kq, kr = unique_klist[pqr]
                ks = kconserv[kp,kq,kr]
                eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
                            (mo_coeff[kp,:,:],mo_coeff[kq,:,:],mo_coeff[kr,:,:],mo_coeff[ks,:,:]),
                            (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
                eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo)
                eri[kp,kq,kr] = eri_kpt.copy()

            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ikp, ikq, ikr = khelper.get_irrVec(kp,kq,kr)
                        irr_eri = eri[ikp,ikq,ikr]
                        eri[kp,kq,kr] = khelper.transform_irr2full(irr_eri,kp,kq,kr)

            # Checking some things...
            maxdiff = 0.0
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp,kq,kr]
                        for p in range(nmo):
                            for q in range(nmo):
                                for r in range(nmo):
                                    for s in range(nmo):
                                        pqrs = eri[kp,kq,kr,p,q,r,s]
                                        rspq = eri[kr,ks,kp,r,s,p,q]
                                        diff = numpy.linalg.norm(pqrs - rspq).real
                                        if diff > 1e-5:
                                            print "** Warning: ERI diff at ",
                                            print "kp,kq,kr,ks,p,q,r,s =", kp, kq, kr, ks, p, q, r, s
                                        maxdiff = max(maxdiff,diff)
            print "Max difference in (pq|rs) - (rs|pq) = %.15g" % maxdiff
            #print "ERI ="
            #print eri

            # Chemist -> physics notation
            eri = eri.transpose(0,2,1,3,5,4,6)

            self.dtype = eri.dtype
            self.oooo = eri[:,:,:,:nocc,:nocc,:nocc,:nocc].copy() / nkpts
            self.ooov = eri[:,:,:,:nocc,:nocc,:nocc,nocc:].copy() / nkpts
            self.ovoo = eri[:,:,:,:nocc,nocc:,:nocc,:nocc].copy() / nkpts
            self.oovv = eri[:,:,:,:nocc,:nocc,nocc:,nocc:].copy() / nkpts
            self.ovov = eri[:,:,:,:nocc,nocc:,:nocc,nocc:].copy() / nkpts
            self.ovvv = eri[:,:,:,:nocc,nocc:,nocc:,nocc:].copy() / nkpts
            self.vvvv = eri[:,:,:,nocc:,nocc:,nocc:,nocc:].copy() / nkpts
            #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)

            # TODO: Avoid this.
            # Store all for now, while DEBUGGING
            self.voov = eri[:,:,:,nocc:,:nocc,:nocc,nocc:].copy() / nkpts
            self.vovo = eri[:,:,:,nocc:,:nocc,nocc:,:nocc].copy() / nkpts
            self.vovv = eri[:,:,:,nocc:,:nocc,nocc:,nocc:].copy() / nkpts
            self.oovo = eri[:,:,:,:nocc,:nocc,nocc:,:nocc].copy() / nkpts
            self.vvov = eri[:,:,:,nocc:,nocc:,:nocc,nocc:].copy() / nkpts
            self.vooo = eri[:,:,:,nocc:,:nocc,:nocc,:nocc].copy() / nkpts

        log.timer('CCSD integral transformation', *cput0)


class _IMDS:
    def __init__(self, cc):
        self.cc = cc

    def make_ip(self):
        cc = self.cc
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        self.Lvv = imdk.Lvv(cc,t1,t2,eris)
        self.Loo = imdk.Loo(cc,t1,t2,eris)
        self.Fov = imdk.cc_Fov(cc,t1,t2,eris)
        self.Wooov = imdk.Wooov(cc,t1,t2,eris)
        self.Wovvo = imdk.Wovvo(cc,t1,t2,eris)
        self.Wovoo = imdk.Wovoo(cc,t1,t2,eris)
        self.Woooo = imdk.Woooo(cc,t1,t2,eris)
        self.Wovov = imdk.Wovov(cc,t1,t2,eris)
        self.Woovv = eris.oovv

    def make_ea(self):
        cc = self.cc
        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        self.Lvv = imdk.Lvv(cc,t1,t2,eris)
        self.Loo = imdk.Loo(cc,t1,t2,eris)
        self.Fov = imdk.cc_Fov(cc,t1,t2,eris)
        self.Wvovv = imdk.Wvovv(cc,t1,t2,eris)
        self.Wvvvo = imdk.Wvvvo(cc,t1,t2,eris)
        self.Wovvo = imdk.Wovvo(cc,t1,t2,eris)
        self.Wvvvv = imdk.Wvvvv(cc,t1,t2,eris)
        self.Woovv = eris.oovv
        self.Wovov = imdk.Wovov(cc,t1,t2,eris)


def print_james_header():
    print ""
    print " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "
    print " You are now about to use the kpointified/restrictified version "
    print " of eom-ccsd                                                    "
    print "                                           -James               "
    print " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "
    print ""
    return


def print_james_ascii():
    print "\
NmmmmmmddddddddddddmmmmmNNNNNNNNNNNNNNNNNNmmmmmddddddddddddmmmmmddhhhyyyyyyyyyyyyyyyyyhhmmNmmddmmmmm\n\
mmmmmmmddddddddmmmmmmmmNNNNNNNNNNNNNmmmmddhhhhhhhhhhhdddddddddhhhhyyyyyyyyyyyyyyyyyyyyhhmmmmmmdddddd\n\
mmmmmmmdddddddmmmmmmmmmmmmNNNNNNNNNmmdhysyyyyyhhhhddddddddddhhhyyyyyyyyyyyyyyyyyyyyyyyhhmmmmdddddddm\n\
mmmmmmmmmddddddmmmmmmmmmmmmmmNNNNmddyyssyyyhhhhdddmmmmmmmmmmddddhhhhhhhhhyyyyyyyyyyyyyhhmmmddddddddd\n\
mmmmmmmmmmddddmmmmmmmmmmmmNNNNNmmhysssyyhhhhhddddmmmNmmmdmmmmmmmmmmmmddddhhhyyyyyyyyyyhhmmmddhhddddd\n\
mmmmmmmdddmmmmmmmmmmmmmmmmNNNNmdhyysyyhhhhhdddddmmmmmmmmmmmmmmmmmNNmmmmmmddhhyyyyyyyyyhhmmmmdddddddd\n\
mmdddddmmmddmmmmmmmmmmmmmmNNmddyyyyyhhhhhhhdddmmmmmmmmmmmmmmmmmmmmNmmmmmmmmdhhhyyyyyyyhhmmmmmddddddd\n\
ddhhhddmmmmmmmmmmmmmmmmmmmmdhyyyyyyyhhddhhddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmdhhyyhhhyyhhmmmmdddddddd\n\
ddhddddmmmmmmmmmmmmmmmddddhhyyyyyyhhhhdhhhdddddddddddddddddddddmmmmmmmmmmmmmddhhyyyyyyhhmmmddddddddd\n\
dddddddmmmmmmmmmmmmdddddhhyyyssyyhhdddhhhhhddddhhhhhdhhhhhhyyyhhdddmmmmmmmNNmmdhhhhhyyhhmmmmdddddddd\n\
ddddddmmmmmmmmmmmddddhhhyyyyyyyhhhdddhhhhhhdhhhyyyyyyhhhhhhhhhhhdddmmmmmmmNNmmmddhhhhyhhmmNmmmmmmmmd\n\
ddhhhddmmmmmmmmddddddhhyhhyhhhhhhhhhhhhhhhhhhyyyyyyyyyyyhhhyyyhhhhhddddmmmmmmmmmdddddhhhmmNmmmmmmmmm\n\
hhdhhdmmmmmmdddddddhhhhhyyhhhhhhhhhhhhyhhyyyyyssssssyyyssssoooooooosssyyhhddmmmmmdddddddmmmmmddddddd\n\
yhhhhddmmddmddhhhhhhhhhhyydhhhhhhyyyyyyyyssysssoooossssoo++++++++++++ooossyhddddddddddddmmmmmmdddddd\n\
oossohdmmmddhddddhhdhhhhhhdhhhyyyyysyyyyssssssooosooso+++//////////+++++oossyhddddddddddmmmmmddddddd\n\
...--oyddddhddddhhhhhhhhhhhhyyysssyyyyyhhhyyysooooo++////://////////////++oossyhdddddddddddddddddddd\n\
     +sddddhddhhhddhhhhhhhyyyyyyssyyhhhhhhhysooo++////::::::::::://:://///+++osyydddddmmmmdddhhhdddd\n\
     +odhhhhhhhhdddhhyhhyyyyyyyyysyyhddddhyoo++/////::::::::::::::::::::////++osshhdddddmmmdhhhhhhhh\n\
     /ohhhhhhhhhdddhhyyyssyyhhhyyyyhddddyso+//:---/:-----::::::::::::::://///++ooyyhhdddmmmddhhhhhhh\n\
     /ohhhhhhhhhhddhysssyyhhdddyysyhddys+++//:----:--------::::::::::::://////++ossyyhddmmmddhhhhhhh\n\
    `/ohhhhhhhhhhddyyoosyhdddhhsshhdys///++::.--------------:::::::::::::://///++ooosyhdmmddhhhhhhhh\n\
 ``..+ohhdddhhhhyhhysosshhdddyyyhdys/:////:---------.--------::::::::::::://///++ooooshdmmmddhhhhhhh\n\
`....+ohddddhhhyyyhsssyhdddhhhhhhs+/:://:::-------::--------::::::::::::://///++ossosshdmmmmmhhhhhhh\n\
````.+oddddddhyysyyssyhdddyyhhhso//:://::::------::::-------::::::::::::::////+oossssyhdmmNmmhhhhhhh\n\
--.-:osddddddhyysssssyhdhhsyyss+/::::::::::---------------:::::::::::::://////+oossyyyhdmmNmmddhhhhh\n\
///++yhmmmddhhyyyyysyyhhyyooooo//////::::::::-------------::::--::::://///////+oossyyhddmmNmmddhhhdd\n\
hhhhhddddmdhhhyyyyysyyyyoo++o++/////////://::::-----------------::://+++++oo++ooossyhhddmmmmmddhdddd\n\
ddhhhdddddhhhhyyyyysssso++++++++++oooooooooooo++//:::--------://++oosssyyyyyyyyyysyyhdddddmmmmdddddd\n\
ssooshdmddhyhhyyhyyssso++++oosssssssoossosssssssso++//::---::/++osssssssooooossyyyyyhdhhhhddmddddddd\n\
ooooshdddhhhddhhhhhyyso++++osssoo++++++++++oooooooo++//:::::/+ooooooooo+++++/++osyyyyhhyhhdddhhyyyyy\n\
ooooohdmddhhddhhdhhhyso+++ooo+++++++++oooooooooo+++++//::::/++ooooooooooooooooooossyyhhhddmddhhyyyhh\n\
ssssshddddddddddhhhhyso+++++///++ooooosyhys//++o+++++//:::/+oooo+++++yyhso++sss++oosyhhdmmmmdhhhhhhh\n\
sssssdddddddmmdhyhhhyoo++///::/++ooo++osyo+::/+++/++++/:::/+oooo+//::ssso+++ssso+++oyhddNNNmddhhhhhh\n\
ssssshddmmmddddhhyyhyso+///::::///++//////////++//++++//:://+ooo++////////++++++///+syddNNNmdddddddh\n\
ssosshdmmdddsssydyyhyso+++/::::::::/::::::/://///+++++//:::/++oo++/////////////////+syddNmmddddddddd\n\
ooooohdmddhy//+shyyhyyoo++/:::::::::::::::::::///++++///:::/++ooo++///:::::::://///oshddmmmddddddddd\n\
oosoohdmddyy:::+oyhhhyso++//::::::::::::::::::///+/++////://++oo++///:::::::::::://oshddmmmddddddddd\n\
ooooohdmmdhh//://syhhhyso+//::::::::::-----::::///++++//////++ooo+///:::::::::::://oyhdmmmmdddddddhh\n\
oossshdmmdddo+///ssddhyyoo///::::::--------::://++++++///:::/+oo++//::::::::::::://oyhmmmmmddddddhhh\n\
ooossddmmdmmys///+odhhysoo///:::::----------://++++++/:::-:://+o+++::::::::::::://+oydddmmmdhddddhhh\n\
//+ooddmmdmmhy+//++hhyssoo///:::::----------://++++++//::::://+++++::::---::::::/++syyyhmmmddddddddd\n\
///++yhmmmmmdhoo+//osysooo////:::::-----::-:://++++++++++///+++++++/::----::::://+ossoyymmmddddddddd\n\
/////+ohdmmmddhys++osyssoo+////:::::--:::::////++++++++++++++++++++//::--:::::///+osooyhNmmddddddddd\n\
//////+sydmmmmmddhhddhysoo/////::::-:://:////////+///////////+++++++//:::::::://+ooyyyhdNmmddddddddd\n\
++++++++shdmmmmmmmmNmdhyoo/////:::::://///+++///////:///////////++++++//::::::/++oshhhhdNmmddddddddd\n\
ooooooooshdmmmmmmmmNmmhyoo/////::-:::///+//////////:::///////////++++++/////://++oshhhhdmmmddddddddd\n\
oooooooosydmmmmmmmmmmmdhso+////::--::://++++++++++///////////////++++++////////++sshhhhdmmmddddddhdd\n\
oooooooosydmmmmmmmmmmmdhss++/////:--:::///+oooooooooo++++++++++oooooo+////////+++syhhhhdmmmddddhhhhh\n\
ooooooo+sddmmmmmmmmmmmddyso+/////::---::://++o+++++++////+++++ooooooo////:://++ooyyhhhhdmmddhhhsyyyy\n\
ooooooooymmmmmmmmmmmmNmdysso+++//:::--:::///////////::::::////++++++///:::///+ooohhhhhhdmmddhhhyyhhh\n\
oooooooyhmmmmmmmmmmmmNmdsssso++/////:://////////////:::::///////+++/////////++osyhhhhhddmmmddhhhhdhh\n\
ooooossddmmmmmmmmmmdmmmdsossooo++/////////////:://///////////////////++/++++oosyhhhhhhhdmmmddddddddd\n\
ooooohhmmmmmmmmmmmmmmmdhoo+osss++++///++//////::::::///////////////++++++++osyyyyhhhyyhdmmdhyhhhhhhh\n\
ooossdmmmmmmmmmmmmmdhsoooo/+oooooo++++++////////::::::///////////+++++oooooosyyyyhhyyyhdmdhyshhyyyyy\n\
oossydmddmmmmmdmmddo+:+ooo///++oosoo+++++++///////:::::::://////++++++osssso//+yhhhhhhhhmmdhhhhyyyyy\n\
ssyyydmmmmmmmmmmdys+++syo++////++oooooooo+++//////::::::://///++++ooosssssso::/oshhhhhhhmmmdhdhyyyyy\n\
yyyyydmmmmmdmmmdysooossso+////////+oossoooo++++///////////////+++ooosssooooooo++oyyhhyhhmmmdhddhhhhh\n\
yyyyydmmmmmmmmdyo++osyss++//////////+ooooooooo+++/////+++++++++ooosssoo++ooosso+/ssyyyhhmmmddddhdddd\n\
yyyyydmmmmmmmmdy+++oosss++///////:////++++oooooooooo+oooooooooossssso+++++ooyss++ooyyyhhmmmddddddddd\n\
yyyyymmmmdddmdhs///ooooo///////::::///+///++++++oossssssssoooooossso+//+++oossso++osyyhhmmmddddddddd\n\
    "
    return


def print_tim_ascii():
    print "\n\
                                       .-/+ossyyyyssssooo+/-.`                                      \n\
                                 ./oydmmmmmdhddhhhddddhdmNNNNNmyo:`                                 \n\
                             -+ymNNNNmmddhhhdhyhhhhhhdmNNNmmNNNNNdso/:-                             \n\
                         `-omNMMNNmmmdhddddddmmNNNNNNMMMMNNNNNNNmmysosmNy/`                         \n\
                       -+syymmmdmmmddddmNNNNNmmNMMMMMMMMMMMMMMNNNmdyssdMMNNy:                       \n\
                    `/yhdddmmmmmmmmddmmNNMMNmmmNMMMMMMMMMMMMMMMMMNNmdyhNMMMMMm+`                    \n\
                  `/shdNNmddddddmmmNNNNNNNNmmmNNNNMNNMMMMMMMMNNNNNNNNmmmNMMMMMMNs.                  \n\
                `+mNNNNNmmdddmmddmmmmNNNNNNNmNNNNNNNNMMMMMNNNNNNNNNNNNNmmNNMMMMMMNo`                \n\
               .sNMMNNNmmddddmmmmddddmmmNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNmmmNMMMMMMMm:               \n\
             `/oyNMMNNmddhdmmddmddddmmmmNNNNNNNNNNNNNNNNNNNmmNNNNNNNNNmmNmmNMMMMMMMMNs`             \n\
            .sysymMMmdyyhhhdmmmmddmmmNNNmmmmNmmNmmmmmdmdddddmmNNNNNNNmmmmNmmNMMMMMMMMMd-            \n\
           -mdysydMMmhyyyhddddmmmmmmmNNNmmddddddmmmmhysssssshmNNNNNNmNmmmmmmNMMMMMMMMMMm:           \n\
          -mNmhyyhMMmhyyyhddddddmmmmmmdhyssoooosyyyyoo++++oooydmmmmmmmNmmmmNNNMMMMMMMMMMN:          \n\
         .hmdddhhydyshhhdddddhhddddhyso+++////++++o++++++++++oyddmmmmmmNmmNNNNMMMMMMMMMMMN-         \n\
        `sddhso+/:::+yydddddhyhysoo+++/////////://////////++/+oshdmmmmmmmmmNNNMMMMMMMMMMMMm`        \n\
        /hhhyso+//:::/sdmmmhyss+//+///://:::::-:://///////////+oshdmmmmmmmNNNNMMMMMMMMMMMMMy        \n\
       .hdddhhhhdddhysydmmmyso+/////::::::::::::::::::////////++oyhmmmmmNNNNNNdNMMMMMMMMMMMN:       \n\
       smddddhhhdmmdddddmmmyo+//////:::::::::::::::::::////////++oydmmmmNNNNdyosdMMMMMMMMMMMh       \n\
      .mmmmmmddyyyyyyyhhhmms++////////:::::::::::::::::://///////++shmmNNNmdyoooyMMMMMMMMMMMN.      \n\
      :NNNNNNmmysoooooosyddo++////::::::::::::----::::::///////////+ohmNNmdyo+++yMMMMMMMMMMMM+      \n\
      oNmmmmmmmmdhhysssyydms++//:::---:::::::::-:://++ooo++/////////+shmNmys+o++hMMMMMMMMMMMMy      \n\
      ymmmmmmmmmmmNmmmmmmmmy++/:::::::::::://+osyhhhyyssssoo++//////+oyhmd+o/+/omMMMMMMMMMMMMd      \n\
      ymmmmmmmdddhdNNNNNNNNmoooosssoo++//++osyyhddddyyyyysoo++++++/+++oydh/+//+yMMMMMMMMMMMMMd      \n\
      sddhhyssoooohNNNNmmmdNyhhhhhddhyysso+ossyyhdddyssysso++//+++++++oydh+//+sNMMMMMMMMMNNNNh      \n\
      :oo++++++++odNMMNmdhhmmssshdmmyoshho:/ossssssssooso+//:://///+++osyyo+sdNMMMMMMMMNNMMMMs      \n\
      .+++++++oooshNMMMmdmNNNsosyyyysssss/::/+oooo++++++/::::::////++ooosyyymMMMMMMMMMMMMMMMM/      \n\
       +ossyhddmmdmMMMMMNMNMNdo+++oooo++o/:::///++/////:::--::////+++oossyyymMMMMMMMMNNNNNMMN`      \n\
       +mmmdhyssoohNMMMMMMMMMNo////////++/:::///+++ooo++//::////+++++oosyysydNNMMMMMMMmNNNMMs       \n\
       `oooo+++++odMMMMMMMMNNNy//::///+o+/::///////oooooooo++++++++++oosyyssyyhdmMMMMMMNNMMm.       \n\
        .+oooooosydMMMNNmdhyyNd////+++o+////://++osso++++ooooo++++++oosyyyssyo:/sdNMMMMMMMN+        \n\
         /yyhdmNNNmddhyyssssymNs++++oosso+++++sssso+/+++oosysso+++++ossyyysssy//::/sdmmmNNy         \n\
         `hNNmdhyyysyysssssssyhy++ooossssoooosssssooo+sosyhoooo++++oossyyoosss/://:-.-///+`         \n\
          `osssssssssssssssssssssoooosssssyhsssoo///+osyys+++oo++ooosyyyoooss+-:://::-.--           \n\
           `/ssssooosssssyyyyyyyhyo++ooosshdhso+ooossysso++++o+oosssyyyoooosso:-::/::::.            \n\
             :ossssyyyyyyyyhhhhhhhyo++++ooosyysyssssoo+++++++oosssyyhysoooosso::-::/::.             \n\
              -oyyyyyyhhhhhhhhhhhy+--:+oo+++oossssoo++++++++oosssyyyysoooooooo::::-::.              \n\
               `/yhhhhhhhhhhhhhho----``-/+++++++++++/////+++osyyyyyysoo+ooooso+-::-.                \n\
                 .ohhhhhhhhhyo+:.---:.```.-++++/////////+oosyyhyysssooooooooooo--.`                 \n\
                   .ohhhs+:......-:--.`````/oo++//++++++osyhhyyssoooooooooooooo-`                   \n\
                     .-.....``..------..`.`.+yooooossssyyhhyyssoooooooooosoo+:`                     \n\
                         ````..--------..`..-sdyssssyssssssssooooooooo++oo/.                        \n\
                            ``.:-:-.-.---..``:sdysooooooooooosssoooo+++:-`                          \n\
                               `.----..---.`-//+so+osoo++++++ooooo+:-`                              \n\
                                   ```...-./o//::--:/yyo++++//:-.`                                  \n\
                                           .....```.`.:.```                                         "
    return
