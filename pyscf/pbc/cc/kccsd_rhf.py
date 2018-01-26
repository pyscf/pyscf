#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import time
import tempfile
import numpy
import numpy as np
import h5py

from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.pbc.cc.kccsd import get_moidx
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.lib import linalg_helper
from pyscf.pbc.lib import kpts_helper

#einsum = np.einsum
einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata, et al., J. Chem. Phys. 120, 2581 (2004)

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
        nocc = cc.nocc
        nvir = cc.nmo - nocc
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
        t1new, t2new = cc.update_amps(t1, t2, eris)
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


def update_amps(cc, t1, t2, eris):
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

    kconserv = cc.khelper.kconserv

    Foo = imdk.cc_Foo(t1,t2,eris,kconserv)
    Fvv = imdk.cc_Fvv(t1,t2,eris,kconserv)
    Fov = imdk.cc_Fov(t1,t2,eris,kconserv)
    Loo = imdk.Loo(t1,t2,eris,kconserv)
    Lvv = imdk.Lvv(t1,t2,eris,kconserv)
    Woooo = imdk.cc_Woooo(t1,t2,eris,kconserv)
    Wvvvv = imdk.cc_Wvvvv(t1,t2,eris,kconserv)
    Wvoov = imdk.cc_Wvoov(t1,t2,eris,kconserv)
    Wvovo = imdk.cc_Wvovo(t1,t2,eris,kconserv)

    # Move energy terms to the other side
    for k in range(nkpts):
        Foo[k] -= np.diag(np.diag(foo[k]))
        Fvv[k] -= np.diag(np.diag(fvv[k]))
        Loo[k] -= np.diag(np.diag(foo[k]))
        Lvv[k] -= np.diag(np.diag(fvv[k]))

    # T1 equation
    t1new = np.array(fov).astype(t1.dtype).conj()
    for ka in range(nkpts):
        ki = ka
        # kc == ki; kk == ka
        t1new[ka] += -2.*einsum('kc,ka,ic->ia',fov[ki],t1[ka],t1[ki])
        t1new[ka] += einsum('ac,ic->ia',Fvv[ka],t1[ki])
        t1new[ka] += -einsum('ki,ka->ia',Foo[ki],t1[ka])

        tau_term = np.empty((nkpts,nocc,nocc,nvir,nvir), dtype=t1.dtype)
        for kk in range(nkpts):
            tau_term[kk] = 2*t2[kk,ki,kk] - t2[ki,kk,kk].transpose(1,0,2,3)
        tau_term[ka] += einsum('ic,ka->kica',t1[ki],t1[ka])

        for kk in range(nkpts):
            kc = kk
            t1new[ka] += einsum('kc,kica->ia',Fov[kc],tau_term[kk])

            t1new[ka] +=  einsum('akic,kc->ia',2*eris.voov[ka,kk,ki],t1[kc])
            t1new[ka] +=  einsum('kaic,kc->ia', -eris.ovov[kk,ka,ki],t1[kc])

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
    t2new = np.array(eris.oovv).conj()
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
                for a in range(nvir):
                    Wvvvv_a = np.array(Wvvvv[ka,kb,kc,a])
                    t2new[ki,kj,ka,:,:,a,:] += einsum('bcd,ijcd->ijb',Wvvvv_a,tau_term)

            t2new[ki,kj,ka] += einsum('ac,ijcb->ijab',Lvv[ka],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('bc,jica->ijab',Lvv[kb],t2[kj,ki,kb])

            t2new[ki,kj,ka] += einsum('ki,kjab->ijab',-Loo[ki],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('kj,kiba->ijab',-Loo[kj],t2[kj,ki,kb])

            kc = kconserv[ka,ki,kb]
            tmp2 = np.array(eris.vovv[kc,ki,kb]).transpose(3,2,1,0).conj() \
                    - einsum('kbic,ka->abic',eris.ovov[ka,kb,ki],t1[ka])
            tmp  = einsum('abic,jc->ijab',tmp2,t1[kj])
            t2new[ki,kj,ka] += tmp
            #P(ij)P(ab)
            kc = kconserv[kb,kj,ka]
            tmp2 = np.array(eris.vovv[kc,kj,ka]).transpose(3,2,1,0).conj() \
                    - einsum('kajc,kb->bajc',eris.ovov[kb,ka,kj],t1[kb])
            tmp  = einsum('bajc,ic->ijab',tmp2,t1[ki])
            t2new[ki,kj,ka] += tmp

            # ka - ki + kk = kj
            # => kk = ki - ka + kj
            kk = kconserv[ki,ka,kj]
            tmp2 = np.array(eris.ooov[kj,ki,kk]).transpose(3,2,1,0).conj() \
                    + einsum('akic,jc->akij',eris.voov[ka,kk,ki],t1[kj])
            tmp  = einsum('akij,kb->ijab',tmp2,t1[kb])
            t2new[ki,kj,ka] -= tmp
            #P(ij)P(ab)
            kk = kconserv[kj,kb,ki]
            tmp2 = np.array(eris.ooov[ki,kj,kk]).transpose(3,2,1,0).conj() \
                    + einsum('bkjc,ic->bkji',eris.voov[kb,kk,kj],t1[ki])
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
            eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2new[ki,kj,ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    fock = eris.fock
    e = 0.0 + 1j*0.0
    for ki in range(nkpts):
        e += 2*einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki])
    t1t1 = numpy.zeros(shape=t2.shape, dtype=t2.dtype)
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


def get_nocc(cc):
    '''The number of occupied orbitals per k-point.'''
    if cc._nocc is not None:
        return cc._nocc
    elif isinstance(cc.frozen, (int, numpy.integer)):
        nocc = int(cc.mo_occ[0].sum()) // 2 - cc.frozen
    elif isinstance(cc.frozen[0], (int, numpy.integer)):
        occ_idx = cc.mo_occ[0] > 0
        occ_idx[list(cc.frozen)] = False
        nocc = numpy.count_nonzero(occ_idx)
    else:
        raise NotImplementedError
    return nocc

def get_nmo(cc):
    '''The number of molecular orbitals per k-point.'''
    if cc._nmo is not None:
        return cc._nmo
    if isinstance(cc.frozen, (int, numpy.integer)):
        nmo = len(cc.mo_occ[0]) - cc.frozen
    elif isinstance(cc.frozen[0], (int, numpy.integer)):
        nmo = len(cc.mo_occ[0]) - len(cc.frozen)
    else:
        raise NotImplementedError
    return nmo


class RCCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.nkpts = len(self.kpts)
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.made_ee_imds = False
        self.made_ip_imds = False
        self.made_ea_imds = False

    nocc = property(get_nocc)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    nmo = property(get_nmo)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=eris.fock.dtype)
        t2 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.fock.dtype)
        woovv = numpy.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.fock.dtype)
        self.emp2 = 0
        foo = eris.fock[:,:nocc,:nocc].copy()
        fvv = eris.fock[:,nocc:,nocc:].copy()
        eris_oovv = np.array(eris.oovv).copy()
        eia = numpy.zeros((nocc,nvir))
        eijab = numpy.zeros((nocc,nocc,nvir,nvir))

        kconserv = self.khelper.kconserv
        for ki in range(nkpts):
          for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                eia = np.diagonal(foo[ki]).reshape(-1,1) - np.diagonal(fvv[ka])
                ejb = np.diagonal(foo[kj]).reshape(-1,1) - np.diagonal(fvv[kb])
                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                woovv[ki,kj,ka] = (2*eris_oovv[ki,kj,ka] - eris_oovv[ki,kj,kb].transpose(0,1,3,2))
                t2[ki,kj,ka] = eris_oovv[ki,kj,ka] / eijab

        t2 = numpy.conj(t2)
        self.emp2 = numpy.einsum('pqrijab,pqrijab',t2,woovv).real
        self.emp2 /= nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        self.dump_flags()
        if eris is None:
            #eris = self.ao2mo()
            eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
        else:
            cctyp = 'CCSD'
            self.converged, self.e_corr, self.t1, self.t2 = \
                    kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                           tol=self.conv_tol,
                           tolnormt=self.conv_tol_normt,
                           max_memory=self.max_memory, verbose=self.verbose)
            if self.converged:
                logger.info(self, 'CCSD converged')
            else:
                logger.info(self, 'CCSD not converged')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.info(self, 'E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)

    def ipccsd(self, nroots=1, koopmans=False, guess=None, partition=None,
               kptlist=None):
        '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested per k-point
            koopmans : bool
                Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            kptlist : list
                List of k-point indices for which eigenvalues are requested.
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        size = nocc + nkpts*nkpts*nocc*nocc*nvir
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ip_partition = partition
        if kptlist is None:
            kptlist = range(nkpts)
        evals = np.zeros((len(kptlist),nroots), np.float)
        evecs = np.zeros((len(kptlist),nroots,size), np.complex)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            if partition == 'full':
                self._ipccsd_diag_matrix2 = self.vector_to_amplitudes_ip(self.ipccsd_diag())[1]

            adiag = self.ipccsd_diag()
            user_guess = False
            if guess:
                user_guess = True
                assert len(guess) == nroots
                for g in guess:
                    assert g.size == size
            else:
                guess = []
                if koopmans:
                    for n in range(nroots):
                        g = np.zeros(size)
                        g[self.nocc-n-1] = 1.0
                        guess.append(g)
                else:
                    idx = adiag.argsort()[:nroots]
                    for i in idx:
                        g = np.zeros(size)
                        g[i] = 1.0
                        guess.append(g)

            def precond(r, e0, x0):
                return r/(e0-adiag+1e-12)

            eig = linalg_helper.eig
            if user_guess or koopmans:
                def pickeig(w, v, nr, envs):
                    x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                    idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                    return w[idx].real, v[:,idx].real, idx
                evals_k, evecs_k = eig(self.ipccsd_matvec, guess, precond, pick=pickeig,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)
            else:
                evals_k, evecs_k = eig(self.ipccsd_matvec, guess, precond,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k

            if nroots == 1:
                evals_k, evecs_k = [evals_k], [evecs_k]
            for n, en, vn in zip(range(nroots), evals_k, evecs_k):
                logger.info(self, 'IP root %d E = %.16g  qpwt = %0.6g',
                            n, en, np.linalg.norm(vn[:self.nocc])**2)
        log.timer('IP-CCSD', *cput0)
        self.eip = evals
        return self.eip, evecs

    def ipccsd_matvec(self, vector):
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

        t1,t2 = self.t1, self.t2
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.khelper.kconserv

        # 1h-1h block
        Hr1 = -einsum('ki,k->i',imds.Loo[kshift],r1)
        # 1h-2h1p block
        for kl in range(nkpts):
            Hr1 += 2.*einsum('ld,ild->i',imds.Fov[kl],r2[kshift,kl])
            Hr1 +=   -einsum('ld,lid->i',imds.Fov[kl],r2[kl,kshift])
            for kk in range(nkpts):
                kd = kconserv[kk,kshift,kl]
                Hr1 += -2.*einsum('klid,kld->i',imds.Wooov[kk,kl,kshift],r2[kk,kl])
                Hr1 +=     einsum('lkid,kld->i',imds.Wooov[kl,kk,kshift],r2[kk,kl])

        Hr2 = np.zeros(r2.shape, dtype=t1.dtype)
        # 2h1p-1h block
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                Hr2[ki,kj] -= einsum('kbij,k->ijb',imds.Wovoo[kshift,kb,ki],r1)
        # 2h1p-2h1p block
        if self.ip_partition == 'mp':
            nkpts, nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:,:nocc,:nocc]
            fvv = fock[:,nocc:,nocc:]
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki,kshift,kj]
                    Hr2[ki,kj] += einsum('bd,ijd->ijb',fvv[kb],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('li,ljb->ijb',foo[ki],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('lj,ilb->ijb',foo[kj],r2[ki,kj])
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2*r2
        else:
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki,kshift,kj]
                    Hr2[ki,kj] += einsum('bd,ijd->ijb',imds.Lvv[kb],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('li,ljb->ijb',imds.Loo[ki],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('lj,ilb->ijb',imds.Loo[kj],r2[ki,kj])
                    for kl in range(nkpts):
                        kk = kconserv[ki,kl,kj]
                        Hr2[ki,kj] += einsum('klij,klb->ijb',imds.Woooo[kk,kl,ki],r2[kk,kl])
                        kd = kconserv[kl,kj,kb]
                        Hr2[ki,kj] += 2.*einsum('lbdj,ild->ijb',imds.Wovvo[kl,kb,kd],r2[ki,kl])
                        Hr2[ki,kj] += -einsum('lbdj,lid->ijb',imds.Wovvo[kl,kb,kd],r2[kl,ki])
                        Hr2[ki,kj] += -einsum('lbjd,ild->ijb',imds.Wovov[kl,kb,kj],r2[ki,kl]) #typo in Ref
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
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kshift = self.kshift
        kconserv = self.khelper.kconserv

        Hr1 = -np.diag(imds.Loo[kshift])

        Hr2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir), dtype=t1.dtype)
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                for i in range(nocc):
                    for j in range(nocc):
                        for b in range(nvir):
                            if self.ip_partition == 'mp':
                                fock = self.eris.fock
                                foo = fock[:,:nocc,:nocc]
                                fvv = fock[:,nocc:,nocc:]
                                Hr2[ki,kj,i,j,b] = fvv[kb,b,b]
                                Hr2[ki,kj,i,j,b] -= foo[ki,i,i]
                                Hr2[ki,kj,i,j,b] -= foo[kj,j,j]
                            else:
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
        nocc = self.nocc
        nvir = self.nmo - nocc
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
        nocc = self.nocc
        nvir = self.nmo - nocc
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

    def eaccsd(self, nroots=1, koopmans=False, guess=None, partition=None,
               kptlist=None):
        '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

        Kwargs:
            See ipccd()
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        size =  nvir + nkpts*nkpts*nocc*nvir*nvir
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ea_partition = partition
        if kptlist is None:
            kptlist = range(nkpts)
        evals = np.zeros((len(kptlist),nroots), np.float)
        evecs = np.zeros((len(kptlist),nroots,size), np.complex)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            if partition == 'full':
                self._eaccsd_diag_matrix2 = self.vector_to_amplitudes_ea(self.eaccsd_diag())[1]

            adiag = self.eaccsd_diag()
            user_guess = False
            if guess:
                user_guess = True
                assert len(guess) == nroots
                for g in guess:
                    assert g.size == size
            else:
                guess = []
                if koopmans:
                    for n in range(nroots):
                        g = np.zeros(size)
                        g[n] = 1.0
                        guess.append(g)
                else:
                    idx = adiag.argsort()[:nroots]
                    for i in idx:
                        g = np.zeros(size)
                        g[i] = 1.0
                        guess.append(g)

            def precond(r, e0, x0):
                return r/(e0-adiag+1e-12)

            eig = linalg_helper.eig
            if user_guess or koopmans:
                def pickeig(w, v, nr, envs):
                    x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                    idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                    return w[idx].real, v[:,idx].real, idx
                evals_k, evecs_k = eig(self.eaccsd_matvec, guess, precond, pick=pickeig,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)
            else:
                evals_k, evecs_k = eig(self.eaccsd_matvec, guess, precond,
                                       tol=self.conv_tol, max_cycle=self.max_cycle,
                                       max_space=self.max_space, nroots=nroots, verbose=self.verbose)

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k

            if nroots == 1:
                evals_k, evecs_k = [evals_k], [evecs_k]
            nvir = self.nmo - self.nocc
            for n, en, vn in zip(range(nroots), evals_k, evecs_k):
                logger.info(self, 'EA root %d E = %.16g  qpwt = %0.6g',
                            n, en, np.linalg.norm(vn[:nvir])**2)
        log.timer('EA-CCSD', *cput0)
        self.eea = evals
        return self.eea, evecs

    def eaccsd_matvec(self, vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

        t1,t2 = self.t1, self.t2
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.khelper.kconserv

        # Eq. (30)
        # 1p-1p block
        Hr1 = einsum('ac,c->a',imds.Lvv[kshift],r1)
        # 1p-2p1h block
        for kl in range(nkpts):
            Hr1 += 2.*einsum('ld,lad->a',imds.Fov[kl],r2[kl,kshift])
            Hr1 +=   -einsum('ld,lda->a',imds.Fov[kl],r2[kl,kl])
            for kc in range(nkpts):
                kd = kconserv[kshift,kc,kl]
                Hr1 +=  2.*einsum('alcd,lcd->a',imds.Wvovv[kshift,kl,kc],r2[kl,kc])
                Hr1 +=    -einsum('aldc,lcd->a',imds.Wvovv[kshift,kl,kd],r2[kl,kc])

        # Eq. (31)
        # 2p1h-1p block
        Hr2 = np.zeros(r2.shape, dtype=t1.dtype)
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                Hr2[kj,ka] += einsum('abcj,c->jab',imds.Wvvvo[ka,kb,kshift],r1)
        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            nkpts, nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:,:nocc,:nocc]
            fvv = fock[:,nocc:,nocc:]
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift,ka,kj]
                    Hr2[kj,ka] -= einsum('lj,lab->jab',foo[kj],r2[kj,ka])
                    Hr2[kj,ka] += einsum('ac,jcb->jab',fvv[ka],r2[kj,ka])
                    Hr2[kj,ka] += einsum('bd,jad->jab',fvv[kb],r2[kj,ka])
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift,ka,kj]
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
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kshift = self.kshift
        kconserv = self.khelper.kconserv

        Hr1 = np.diag(imds.Lvv[kshift])

        Hr2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir), dtype=t1.dtype)
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            if self.ea_partition == 'mp':
                                fock = self.eris.fock
                                foo = fock[:,:nocc,:nocc]
                                fvv = fock[:,nocc:,nocc:]
                                Hr2[kj,ka,j,a,b] -= foo[kj,j,j]
                                Hr2[kj,ka,j,a,b] += fvv[ka,a,a]
                                Hr2[kj,ka,j,a,b] += fvv[kb,b,b]
                            else:
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
        nocc = self.nocc
        nvir = self.nmo - nocc
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
        nocc = self.nocc
        nvir = self.nmo - nocc
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

    def amplitudes_to_vector(self, t1, t2):
        return np.hstack((t1.ravel(), t2.ravel()))

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        nov = nkpts*nocc*nvir
        t1 = vec[:nov].reshape(nkpts,nocc,nvir)
        t2 = vec[nov:].reshape(nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir)
        return t1, t2


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        moidx = get_moidx(cc)
        nkpts = cc.nkpts
        nmo = cc.nmo

        nao = cc.mo_coeff[0].shape[0]
        dtype = cc.mo_coeff[0].dtype
        self.mo_coeff = numpy.zeros((nkpts,nao,nmo), dtype=dtype)
        self.fock = numpy.zeros((nkpts,nmo,nmo), dtype=dtype)
        if mo_coeff is None:
            for kp in range(nkpts):
                self.mo_coeff[kp] = cc.mo_coeff[kp][:,moidx[kp]]
            mo_coeff = self.mo_coeff
            for kp in range(nkpts):
                self.fock[kp] = numpy.diag(cc.mo_energy[kp][moidx[kp]]).astype(dtype)
        else:  # If mo_coeff is not canonical orbital
            for kp in range(nkpts):
                self.mo_coeff[kp] = mo_coeff[kp][:,moidx[kp]]
            mo_coeff = self.mo_coeff
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            # Don't use get_veff(), because cc._scf might be DFT,
            # but veff should be Fock, not Kohn-Sham.
            #fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            vj, vk = cc._scf.get_jk(cc.mol, dm)
            veff = vj - vk * .5
            fockao = cc._scf.get_hcore() + veff
            for kp in range(nkpts):
                self.fock[kp] = reduce(numpy.dot, (mo_coeff[kp].T.conj(), fockao[kp], mo_coeff[kp])).astype(dtype)

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts, nocc, nvir)
        mem_now = lib.current_memory()[0]
        fao2mo = cc._scf.with_df.ao2mo

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory)
            or cc.mol.incore_anyway):
            log.info('using incore ERI storage')
            self.oooo = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
            self.ooov = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
            self.oovv = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
            self.ovov = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
            self.voov = numpy.zeros((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
            self.vovv = numpy.zeros((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)
            self.vvvv = numpy.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=dtype)

            for (ikp,ikq,ikr) in khelper.symm_map.keys():
                iks = kconserv[ikp,ikq,ikr]
                eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                                 (cc.kpts[ikp],cc.kpts[ikq],cc.kpts[ikr],cc.kpts[iks]), compact=False)
                if dtype == np.float: eri_kpt = eri_kpt.real
                eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo)
                for (kp,kq,kr) in khelper.symm_map[(ikp,ikq,ikr)]:
                    eri_kpt_symm = khelper.transform_symm(eri_kpt,kp,kq,kr).transpose(0,2,1,3)
                    self.oooo[kp,kr,kq] = eri_kpt_symm[:nocc,:nocc,:nocc,:nocc] / nkpts
                    self.ooov[kp,kr,kq] = eri_kpt_symm[:nocc,:nocc,:nocc,nocc:] / nkpts
                    self.oovv[kp,kr,kq] = eri_kpt_symm[:nocc,:nocc,nocc:,nocc:] / nkpts
                    self.ovov[kp,kr,kq] = eri_kpt_symm[:nocc,nocc:,:nocc,nocc:] / nkpts
                    self.voov[kp,kr,kq] = eri_kpt_symm[nocc:,:nocc,:nocc,nocc:] / nkpts
                    self.vovv[kp,kr,kq] = eri_kpt_symm[nocc:,:nocc,nocc:,nocc:] / nkpts
                    self.vvvv[kp,kr,kq] = eri_kpt_symm[nocc:,nocc:,nocc:,nocc:] / nkpts

            self.dtype = dtype
        else:
            log.info('using HDF5 ERI storage')
            _tmpfile1 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.feri1 = h5py.File(_tmpfile1.name)

            self.oooo = self.feri1.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype.char)
            self.ooov = self.feri1.create_dataset('ooov', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype.char)
            self.oovv = self.feri1.create_dataset('oovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype.char)
            self.ovov = self.feri1.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype.char)
            self.voov = self.feri1.create_dataset('voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype.char)
            self.vovv = self.feri1.create_dataset('vovv', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype.char)
            self.vvvv = self.feri1.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype.char)

            # <ij|pq>  = (ip|jq)
            cput1 = time.clock(), time.time()
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp,kq,kr]
                        orbo_p = mo_coeff[kp,:,:nocc]
                        orbo_r = mo_coeff[kr,:,:nocc]
                        buf_kpt = fao2mo((orbo_p,mo_coeff[kq,:,:],orbo_r,mo_coeff[ks,:,:]),
                                         (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)
                        if mo_coeff.dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape(nocc,nmo,nocc,nmo).transpose(0,2,1,3)
                        self.dtype = buf_kpt.dtype
                        self.oooo[kp,kr,kq,:,:,:,:] = buf_kpt[:,:,:nocc,:nocc] / nkpts
                        self.ooov[kp,kr,kq,:,:,:,:] = buf_kpt[:,:,:nocc,nocc:] / nkpts
                        self.oovv[kp,kr,kq,:,:,:,:] = buf_kpt[:,:,nocc:,nocc:] / nkpts
            cput1 = log.timer_debug1('transforming oopq', *cput1)

            # <ia|pq> = (ip|aq)
            cput1 = time.clock(), time.time()
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp,kq,kr]
                        orbo_p = mo_coeff[kp,:,:nocc]
                        orbv_r = mo_coeff[kr,:,nocc:]
                        buf_kpt = fao2mo((orbo_p,mo_coeff[kq,:,:],orbv_r,mo_coeff[ks,:,:]),
                                         (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)
                        if mo_coeff.dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape(nocc,nmo,nvir,nmo).transpose(0,2,1,3)
                        self.ovov[kp,kr,kq,:,:,:,:] = buf_kpt[:,:,:nocc,nocc:] / nkpts
                        self.vovv[kr,kp,ks,:,:,:,:] = buf_kpt[:,:,nocc:,nocc:].transpose(1,0,3,2) / nkpts
                        self.voov[kr,kp,ks,:,:,:,:] = buf_kpt[:,:,nocc:,:nocc].transpose(1,0,3,2) / nkpts
            cput1 = log.timer_debug1('transforming ovpq', *cput1)

            ## Without k-point symmetry
            #cput1 = time.clock(), time.time()
            #for kp in range(nkpts):
            #    for kq in range(nkpts):
            #        for kr in range(nkpts):
            #            ks = kconserv[kp,kq,kr]
            #            orbv_p = mo_coeff[kp,:,nocc:]
            #            orbv_q = mo_coeff[kq,:,nocc:]
            #            orbv_r = mo_coeff[kr,:,nocc:]
            #            orbv_s = mo_coeff[ks,:,nocc:]
            #            for a in range(nvir):
            #                orbva_p = orbv_p[:,a].reshape(-1,1)
            #                buf_kpt = fao2mo((orbva_p,orbv_q,orbv_r,orbv_s),
            #                                 (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)
            #                if mo_coeff.dtype == np.float: buf_kpt = buf_kpt.real
            #                buf_kpt = buf_kpt.reshape((1,nvir,nvir,nvir)).transpose(0,2,1,3)
            #                self.vvvv[kp,kr,kq,a,:,:,:] = buf_kpt[:] / nkpts
            #cput1 = log.timer_debug1('transforming vvvv', *cput1)

            cput1 = time.clock(), time.time()
            for (ikp,ikq,ikr) in khelper.symm_map.keys():
                iks = kconserv[ikp,ikq,ikr]
                orbv_p = mo_coeff[ikp,:,nocc:]
                orbv_q = mo_coeff[ikq,:,nocc:]
                orbv_r = mo_coeff[ikr,:,nocc:]
                orbv_s = mo_coeff[iks,:,nocc:]
                mem_now = lib.current_memory()[0]
                if nvir**4 * 16 / 1e6 + mem_now < cc.max_memory:
                    # unit cell is small enough to handle vvvv in-core
                    buf_kpt = fao2mo((orbv_p,orbv_q,orbv_r,orbv_s),
                                     (cc.kpts[ikp],cc.kpts[ikq],cc.kpts[ikr],cc.kpts[iks]), compact=False)
                    if dtype == np.float: buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape((nvir,nvir,nvir,nvir))
                    for (kp,kq,kr) in khelper.symm_map[(ikp,ikq,ikr)]:
                        buf_kpt_symm = khelper.transform_symm(buf_kpt,kp,kq,kr).transpose(0,2,1,3)
                        self.vvvv[kp,kr,kq] = buf_kpt_symm / nkpts
                else:
                    for a in range(nvir):
                        orbva_p = orbv_p[:,a].reshape(-1,1)
                        buf_kpt = fao2mo((orbva_p,orbv_q,orbv_r,orbv_s),
                                         (cc.kpts[ikp],cc.kpts[ikq],cc.kpts[ikr],cc.kpts[iks]), compact=False)
                        if mo_coeff.dtype == np.float: buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape((1,nvir,nvir,nvir)).transpose(0,2,1,3)

                        self.vvvv[ikp,ikr,ikq,a,:,:,:] = buf_kpt[0,:,:,:] / nkpts
                        # Store symmetric permutations
                        self.vvvv[ikr,ikp,iks,:,a,:,:] = buf_kpt.transpose(1,0,3,2)[:,0,:,:] / nkpts
                        self.vvvv[ikq,iks,ikp,:,:,a,:] = buf_kpt.transpose(2,3,0,1).conj()[:,:,0,:] / nkpts
                        self.vvvv[iks,ikq,ikr,:,:,:,a] = buf_kpt.transpose(3,2,1,0).conj()[:,:,:,0] / nkpts
            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)

def verify_eri_symmetry(nmo, nkpts, kconserv, eri):
    # Check ERI symmetry
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
                                    print("** Warning: ERI diff at ")
                                    print("kp,kq,kr,ks,p,q,r,s =", kp, kq, kr, ks, p, q, r, s)
                                maxdiff = max(maxdiff,diff)
    print("Max difference in (pq|rs) - (rs|pq) = %.15g" % maxdiff)

imd = imdk
class _IMDS:
    # Identical to molecular rccsd_slow
    def __init__(self, cc):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.eris = cc.eris
        self.kconserv = cc.khelper.kconserv
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv
        self.Loo = imd.Loo(t1,t2,eris,kconserv)
        self.Lvv = imd.Lvv(t1,t2,eris,kconserv)
        self.Fov = imd.cc_Fov(t1,t2,eris,kconserv)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris,kconserv)
        self.Wovvo = imd.Wovvo(t1,t2,eris,kconserv)
        self.Woovv = eris.oovv

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris,kconserv)
        self.Wooov = imd.Wooov(t1,t2,eris,kconserv)
        self.Wovoo = imd.Wovoo(t1,t2,eris,kconserv)
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris,kconserv)
        if ea_partition == 'mp' and not np.any(t1):
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,kconserv)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris,kconserv)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,kconserv,self.Wvvvv)
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError

def _mem_usage(nkpts, nocc, nvir):
    incore = nkpts**3*(nocc+nvir)**4
    # Roughly, factor of two for intermediates and factor of two
    # for safety (temp arrays, copying, etc)
    incore *= 4
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore*16/1e6, outcore*16/1e6, basic*16/1e6


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2]), exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KRCCSD(kmf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.155298393321855)

