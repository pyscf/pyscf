#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import time
import numpy
import numpy as np
import h5py

from functools import reduce
from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.pbc import scf
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nocc, get_nmo
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.lib import linalg_helper
from pyscf.pbc.lib import kpts_helper
from pyscf import __config__

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

    #assert(isinstance(eris, pyscf.cc.ccsd._ChemistsERIs))
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
    if isinstance(cc.diis, lib.diis.DIIS):
        adiis = cc.diis
    elif cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris)
        normt = cc.get_normt_diff(t1, t2, t1new, t2new)
        if cc.iterative_damping < 1.0:
            alpha = cc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = cc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, cc.energy(t1, t2, eris)
        log.info('cycle = %d  E(KCCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('KCCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('KCCSD', *cput0)
    return conv, eccsd, t1, t2

def get_normt_diff(cc, t1, t2, t1new, t2new):
    '''Calculates norm(t1 - t1new) + norm(t2 - t2new).'''
    return numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)

def update_amps(cc, t1, t2, eris):
    time0 = time1 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:,:nocc,nocc:]
    foo = fock[:,:nocc,:nocc]
    fvv = fock[:,nocc:,nocc:]

    kconserv = cc.khelper.kconserv

    Foo = imdk.cc_Foo(t1,t2,eris,kconserv)
    Fvv = imdk.cc_Fvv(t1,t2,eris,kconserv)
    Fov = imdk.cc_Fov(t1,t2,eris,kconserv)
    Loo = imdk.Loo(t1,t2,eris,kconserv)
    Lvv = imdk.Lvv(t1,t2,eris,kconserv)

    # Move energy terms to the other side
    for k in range(nkpts):
        Foo[k] -= np.diag(np.diag(foo[k]))
        Fvv[k] -= np.diag(np.diag(fvv[k]))
        Loo[k] -= np.diag(np.diag(foo[k]))
        Lvv[k] -= np.diag(np.diag(fvv[k]))
    time1 = log.timer_debug1('intermediates', *time1)

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
    time1 = log.timer_debug1('t1', *time1)

    # T2 equation
    t2new = np.empty_like(t2)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        t2new[ki,kj,ka] = eris.oovv[ki,kj,ka].conj()

    mem_now = lib.current_memory()[0]
    if (nocc**4*nkpts**3)*16/1e6 + mem_now < cc.max_memory*.9:
        Woooo = imdk.cc_Woooo(t1, t2, eris, kconserv)
    else:
        fimd = lib.H5TmpFile()
        Woooo = fimd.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), t1.dtype.char)
        Woooo = imdk.cc_Woooo(t1, t2, eris, kconserv, Woooo)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
        kb = kconserv[ki,ka,kj]

        t2new_tmp = np.zeros((nocc,nocc,nvir,nvir), dtype=t2.dtype)
        for kl in range(nkpts):
            kk = kconserv[kj,kl,ki]
            tau_term = t2[kk,kl,ka].copy()
            if kl == kb and kk == ka:
                tau_term += einsum('ic,jd->ijcd',t1[ka],t1[kb])
            t2new_tmp += 0.5 * einsum('klij,klab->ijab',Woooo[kk,kl,ki],tau_term)
        t2new[ki,kj,ka] += t2new_tmp
        t2new[kj,ki,kb] += t2new_tmp.transpose(1,0,3,2)
    Woooo = None
    fimd = None
    time1 = log.timer_debug1('t2 oooo', *time1)

    mem_now = lib.current_memory()[0]
    if (nvir**4*nkpts**3)*16/1e6 + mem_now < cc.max_memory*.9:
        Wvvvv = imdk.cc_Wvvvv(t1, t2, eris, kconserv)
    else:
        fimd = lib.H5TmpFile()
        Wvvvv = fimd.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), t1.dtype.char)
        Wvvvv = imdk.cc_Wvvvv(t1, t2, eris, kconserv, Wvvvv)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki,ka,kj]
        t2new_tmp = np.zeros((nocc,nocc,nvir,nvir), dtype=t2.dtype)
        for kc in range(nkpts):
            kd = kconserv[ka,kc,kb]
            tau_term = t2[ki,kj,kc].copy()
            if ki == kc and kj == kd:
                tau_term += einsum('ic,jd->ijcd',t1[ki],t1[kj])
            t2new_tmp += 0.5 * einsum('abcd,ijcd->ijab',Wvvvv[ka,kb,kc],tau_term)

        t2new_tmp += einsum('ac,ijcb->ijab',Lvv[ka],t2[ki,kj,ka])

        t2new_tmp += einsum('ki,kjab->ijab',-Loo[ki],t2[ki,kj,ka])

        kc = kconserv[ka,ki,kb]
        tmp2 = np.asarray(eris.vovv[kc,ki,kb]).transpose(3,2,1,0).conj() \
                - einsum('kbic,ka->abic',eris.ovov[ka,kb,ki],t1[ka])
        t2new_tmp += einsum('abic,jc->ijab',tmp2,t1[kj])

        kk = kconserv[ki,ka,kj]
        tmp2 = np.asarray(eris.ooov[kj,ki,kk]).transpose(3,2,1,0).conj() \
                + einsum('akic,jc->akij',eris.voov[ka,kk,ki],t1[kj])
        t2new_tmp -= einsum('akij,kb->ijab',tmp2,t1[kb])
        t2new[ki,kj,ka] += t2new_tmp
        t2new[kj,ki,kb] += t2new_tmp.transpose(1,0,3,2)
    Wvvvv = None
    fimd = None
    time1 = log.timer_debug1('t2 vvvv', *time1)

    mem_now = lib.current_memory()[0]
    if (nocc**2*nvir**2*nkpts**3)*16/1e6*2 + mem_now < cc.max_memory*.9:
        Wvoov = imdk.cc_Wvoov(t1, t2, eris, kconserv)
        Wvovo = imdk.cc_Wvovo(t1, t2, eris, kconserv)
    else:
        fimd = lib.H5TmpFile()
        Wvoov = fimd.create_dataset('voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), t1.dtype.char)
        Wvovo = fimd.create_dataset('vovo', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nocc), t1.dtype.char)
        Wvoov = imdk.cc_Wvoov(t1, t2, eris, kconserv, Wvoov)
        Wvovo = imdk.cc_Wvovo(t1, t2, eris, kconserv, Wvovo)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki,ka,kj]
        t2new_tmp = np.zeros((nocc,nocc,nvir,nvir), dtype=t2.dtype)
        for kk in range(nkpts):
            kc = kconserv[ka,ki,kk]
            tmp_voov = 2.*Wvoov[ka,kk,ki] - Wvovo[ka,kk,kc].transpose(0,1,3,2)
            t2new_tmp += einsum('akic,kjcb->ijab',tmp_voov,t2[kk,kj,kc])

            kc = kconserv[ka,ki,kk]
            t2new_tmp -= einsum('akic,kjbc->ijab',Wvoov[ka,kk,ki],t2[kk,kj,kb])

            kc = kconserv[kk,ka,kj]
            t2new_tmp -= einsum('bkci,kjac->ijab',Wvovo[kb,kk,kc],t2[kk,kj,ka])

        t2new[ki,kj,ka] += t2new_tmp
        t2new[kj,ki,kb] += t2new_tmp.transpose(1,0,3,2)
    Wvoov = Wvovo = None
    fimd = None
    time1 = log.timer_debug1('t2 voov', *time1)

    for ki in range(nkpts):
        eia = foo[ki].diagonal()[:,None] - fvv[ki].diagonal()
        # When padding the occupied/virtual arrays, some fock elements will be zero
        eia[abs(eia) < LOOSE_ZERO_TOL] = LARGE_DENOM
        t1new[ki] /= eia

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki,ka,kj]
        eia = np.diagonal(foo[ki]).reshape(-1,1) - np.diagonal(fvv[ka])
        ejb = np.diagonal(foo[kj]).reshape(-1,1) - np.diagonal(fvv[kb])
        eijab = eia[:,None,:,None] + ejb[:,None,:]
        # Due to padding; see above discussion concerning t1new in update_amps()
        eijab[abs(eijab) < LOOSE_ZERO_TOL] = LARGE_DENOM

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
    tau = t1t1 = numpy.zeros(shape=t2.shape, dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki,kj,ka] = einsum('ia,jb->ijab',t1[ki],t1[kj])
    tau += t2
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e += 2*einsum('ijab,ijab', tau[ki,kj,ka], eris.oovv[ki,kj,ka])
                e +=  -einsum('ijab,ijba', tau[ki,kj,ka], eris.oovv[ki,kj,kb])
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real


class RCCSD(pyscf.cc.ccsd.CCSD):

    max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KRCCSD_max_space', 20)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, scf.khf.KSCF))
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.made_ee_imds = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.ip_partition = None
        self.ea_partition = None
        self.max_space = 20

        keys = set(['kpts', 'mo_energy', 'khelper', 'made_ee_imds',
                    'made_ip_imds', 'made_ea_imds', 'ip_partition',
                    'ea_partition', 'max_space'])
        self._keys = self._keys.union(keys)

    @property
    def nkpts(self):
        return len(self.kpts)

    get_normt_diff = get_normt_diff
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def dump_flags(self):
        return pyscf.cc.ccsd.CCSD.dump_flags(self)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=eris.fock.dtype)
        t2 = numpy.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.fock.dtype)
        foo = eris.fock[:,:nocc,:nocc].copy()
        fvv = eris.fock[:,nocc:,nocc:].copy()

        emp2 = 0
        kconserv = self.khelper.kconserv
        touched = numpy.zeros((nkpts,nkpts,nkpts), dtype=bool)
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            if touched[ki,kj,ka]:
                continue

            kb = kconserv[ki,ka,kj]
            eia = foo[ki].diagonal().real[:, None] - fvv[ka].diagonal().real
            ejb = foo[kj].diagonal().real[:, None] - fvv[kb].diagonal().real
            eijab = lib.direct_sum('ia,jb->ijab', eia, ejb)
            # Due to padding; see above discussion concerning t1new in update_amps()
            idx = abs(eijab) < LOOSE_ZERO_TOL
            eijab[idx] = LARGE_DENOM

            eris_ijab = eris.oovv[ki,kj,ka]
            eris_ijba = eris.oovv[ki,kj,kb]
            t2[ki,kj,ka] = eris_ijab.conj() / eijab
            woovv = 2*eris_ijab - eris_ijba.transpose(0,1,3,2)
            emp2 += numpy.einsum('ijab,ijab', t2[ki,kj,ka], woovv)

            if ka != kb:
                eijba = eijab.transpose(0,1,3,2)
                t2[ki,kj,kb] = eris_ijba.conj() / eijba
                woovv = 2*eris_ijba - eris_ijab.transpose(0,1,3,2)
                emp2 += numpy.einsum('ijab,ijab', t2[ki,kj,kb], woovv)

            touched[ki,kj,ka] = touched[ki,kj,kb] = True

        self.emp2 = emp2.real / nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    energy = energy
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2=mbpt2)
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

    def vector_size_ip(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        size = nocc + nkpts**2 * nocc**2 * nvir
        return size

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
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ip()
        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            nfrozen = np.sum(self.mask_frozen_ip(np.zeros(size, dtype=int), const=1))
            nroots = min(nroots, size - nfrozen)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ip_partition = partition
        evals = np.zeros((len(kptlist),nroots), np.float)
        evecs = np.zeros((len(kptlist),nroots,size), np.complex)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            adiag = self.ipccsd_diag(kshift)
            adiag = self.mask_frozen_ip(adiag, const=LARGE_DENOM)
            if partition == 'full':
                self._ipccsd_diag_matrix2 = self.vector_to_amplitudes_ip(adiag)[1]

            user_guess = False
            if guess:
                user_guess = True
                assert len(guess) == nroots
                for g in guess:
                    assert g.size == size
            else:
                guess = []
                if koopmans:
                    foo_kshift = self.eris.fock[kshift,:nocc,:nocc]
                    nonfrozen_idx = np.where(abs(foo_kshift.diagonal()) > LOOSE_ZERO_TOL)[0]
                    for n in nonfrozen_idx[::-1][:nroots]:
                        g = np.zeros(size)
                        g[n] = 1.0
                        g = self.mask_frozen_ip(g, const=0.0)
                        guess.append(g)
                else:
                    idx = adiag.argsort()[:nroots]
                    for i in idx:
                        g = np.zeros(size)
                        g[i] = 1.0
                        g = self.mask_frozen_ip(g, const=0.0)
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

        vector = self.mask_frozen_ip(vector, const=0.0)
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
        vector = self.mask_frozen_ip(vector, const=0.0)
        return vector

    def ipccsd_diag(self, kshift=0):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kconserv = self.khelper.kconserv

        Hr1 = -np.diag(imds.Loo[kshift])

        Hr2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir), dtype=t1.dtype)
        if self.ip_partition == 'mp':
            foo = self.eris.fock[:,:nocc,:nocc]
            fvv = self.eris.fock[:,nocc:,nocc:]
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki,kshift,kj]
                    Hr2[ki,kj]  = fvv[kb].diagonal()
                    Hr2[ki,kj] -= foo[ki].diagonal()[:,None,None]
                    Hr2[ki,kj] -= foo[kj].diagonal()[:,None]
        else:
            idx = np.arange(nocc)
            for ki in range(nkpts):
                for kj in range(nkpts):
                    kb = kconserv[ki,kshift,kj]
                    Hr2[ki,kj]  = imds.Lvv[kb].diagonal()
                    Hr2[ki,kj] -= imds.Loo[ki].diagonal()[:,None,None]
                    Hr2[ki,kj] -= imds.Loo[kj].diagonal()[:,None]

                    if ki == kconserv[ki,kj,kj]:
                        Hr2[ki,kj] += np.einsum('ijij->ij', imds.Woooo[ki,kj,ki])[:,:,None]

                    Hr2[ki,kj] -= np.einsum('jbjb->jb', imds.Wovov[kj,kb,kj])

                    Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj,kb,kb])
                    Hr2[ki,kj] += 2.*Wovvo
                    if ki == kj:  # and i == j
                        Hr2[ki,ki,idx,idx] -= Wovvo

                    Hr2[ki,kj] -= np.einsum('ibib->ib', imds.Wovov[ki,kb,ki])[:,None,:]

                    kd = kconserv[kj,kshift,ki]
                    Hr2[ki,kj] -= 2.*np.einsum('ijcb,jibc->ijb', t2[ki,kj,kshift], imds.Woovv[kj,ki,kd])
                    Hr2[ki,kj] += np.einsum('ijcb,ijbc->ijb', t2[ki,kj,kshift], imds.Woovv[ki,kj,kd])

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

    def mask_frozen_ip(self, vector, const=LARGE_DENOM):
        '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
        r1, r2 = self.vector_to_amplitudes_ip(vector)
        nkpts, nocc, nvir = self.t1.shape
        kconserv = self.khelper.kconserv
        kshift = self.kshift

        fock = self.eris.fock
        foo = fock[:,:nocc,:nocc]
        fvv = fock[:,nocc:,nocc:]
        d0 = np.array(const, dtype=r2.dtype)

        r1_mask_idx = abs(foo[kshift].diagonal()) < LOOSE_ZERO_TOL
        r1[r1_mask_idx] = d0
        for ki in range(nkpts):
            imask_idx = abs(foo[ki].diagonal()) < LOOSE_ZERO_TOL
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                jmask_idx = abs(foo[kj].diagonal()) < LOOSE_ZERO_TOL
                bmask_idx = abs(fvv[kb].diagonal()) < LOOSE_ZERO_TOL

                d0 = const * np.array(1.0, dtype=r2.dtype)
                r2[ki, kj, imask_idx] = d0
                r2[ki, kj, :, jmask_idx] = d0
                r2[ki, kj, :, :, bmask_idx] = d0

        vector = self.amplitudes_to_vector_ip(r1, r2)
        return vector

    def vector_size_ea(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        size = nvir + nkpts**2 * nvir**2 * nocc
        return size

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
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ea()
        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            nfrozen = np.sum(self.mask_frozen_ea(np.zeros(size, dtype=int), const=1))
            nroots = min(nroots, size - nfrozen)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ea_partition = partition
        evals = np.zeros((len(kptlist),nroots), np.float)
        evecs = np.zeros((len(kptlist),nroots,size), np.complex)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            adiag = self.eaccsd_diag(kshift)
            adiag = self.mask_frozen_ea(adiag, const=LARGE_DENOM)
            if partition == 'full':
                self._eaccsd_diag_matrix2 = self.vector_to_amplitudes_ea(adiag)[1]

            user_guess = False
            if guess:
                user_guess = True
                assert len(guess) == nroots
                for g in guess:
                    assert g.size == size
            else:
                guess = []
                if koopmans:
                    fvv_kshift = self.eris.fock[kshift,nocc:,nocc:]
                    nonfrozen_idx = np.where(abs(fvv_kshift.diagonal()) > LOOSE_ZERO_TOL)[0]
                    for n in nonfrozen_idx[:nroots]:
                        g = np.zeros(size)
                        g[n] = 1.0
                        g = self.mask_frozen_ea(g, const=0.0)
                        guess.append(g)
                else:
                    idx = adiag.argsort()[:nroots]
                    for i in idx:
                        g = np.zeros(size)
                        g[i] = 1.0
                        g = self.mask_frozen_ea(g, const=0.0)
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

        vector = self.mask_frozen_ea(vector, const=0.0)
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
        vector = self.mask_frozen_ea(vector, const=0.0)
        return vector

    def eaccsd_diag(self, kshift=0):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kconserv = self.khelper.kconserv

        Hr1 = np.diag(imds.Lvv[kshift])

        Hr2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir), dtype=t2.dtype)
        if self.ea_partition == 'mp':
            foo = self.eris.fock[:,:nocc,:nocc]
            fvv = self.eris.fock[:,nocc:,nocc:]
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift,ka,kj]
                    Hr2[kj,ka] -= foo[kj].diagonal()[:,None,None]
                    Hr2[kj,ka] += fvv[ka].diagonal()[None,:,None]
                    Hr2[kj,ka] += fvv[kb].diagonal()
        else:
            idx = np.eye(nvir, dtype=bool)
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift,ka,kj]
                    Hr2[kj,ka] -= imds.Loo[kj].diagonal()[:,None,None]
                    Hr2[kj,ka] += imds.Lvv[ka].diagonal()[None,:,None]
                    Hr2[kj,ka] += imds.Lvv[kb].diagonal()

                    Hr2[kj,ka] += np.einsum('abab->ab', imds.Wvvvv[ka,kb,ka])

                    Hr2[kj,ka] -= np.einsum('jbjb->jb', imds.Wovov[kj,kb,kj])[:,None,:]
                    Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj,kb,kb])
                    Hr2[kj,ka] += 2.*Wovvo[:,None,:]
                    if ka == kb:
                        for a in range(nvir):
                            Hr2[kj,ka,:,a,a] -= Wovvo[:,a]

                    Hr2[kj,ka] -= np.einsum('jaja->ja', imds.Wovov[kj,ka,kj])[:,:,None]

                    Hr2[kj,ka] -= 2*np.einsum('ijab,ijab->jab', t2[kshift,kj,ka], imds.Woovv[kshift,kj,ka])
                    Hr2[kj,ka] += np.einsum('ijab,ijba->jab', t2[kshift,kj,ka], imds.Woovv[kshift,kj,kb])

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

    def mask_frozen_ea(self, vector, const=LARGE_DENOM):
        '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
        r1, r2 = self.vector_to_amplitudes_ea(vector)
        nkpts, nocc, nvir = self.t1.shape
        kconserv = self.khelper.kconserv
        kshift = self.kshift

        fock = self.eris.fock
        foo = fock[:,:nocc,:nocc]
        fvv = fock[:,nocc:,nocc:]
        d0 = np.array(const, dtype=r2.dtype)

        r1_mask_idx = abs(fvv[kshift].diagonal()) < LOOSE_ZERO_TOL
        r1[r1_mask_idx] = d0
        for kj in range(nkpts):
            jmask_idx = abs(foo[kj].diagonal()) < LOOSE_ZERO_TOL
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                amask_idx = abs(fvv[ka].diagonal()) < LOOSE_ZERO_TOL
                bmask_idx = abs(fvv[kb].diagonal()) < LOOSE_ZERO_TOL

                d0 = const * np.array(1.0, dtype=r2.dtype)
                r2[kj, ka, jmask_idx] = d0
                r2[kj, ka, :, amask_idx] = d0
                r2[kj, ka, :, :, bmask_idx] = d0

        vector = self.amplitudes_to_vector_ea(r1, r2)
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

KRCCSD = RCCSD

def pad_frozen_kpt_mo_coeff(cc, mo_coeff):
    '''Creates the mo_coeff and fock molecular orbital matrix elements for
    frozen orbitals.

    The padded molecular orbital coefficients at each k-point are done so that
    the first `nocc[ikpt]` and last `nvir[ikpt]` molecular orbital coefficients
    of the padded array are the same as those of `mo_coeff` while the remaining
    elements in the array are zero.  Here `nocc[ikpt]` and `nvir[ikpt]`

    Note:
        When remaking the fock matrix elements with the padded arrays, some
        fock matrix elements will be zero.  This needs to be accounted for in
        most algorithms.

    Args:
        cc (:class:`CCSD`): Coupled-cluster object storing results of a coupled-
            cluster calculation.
        mo_coeff (:obj:`ndarray`): Molecular orbital coefficients.

    Returns:
        padded_mo_coeff (:obj:list of `ndarray`): Molecular orbital coefficients
            with padding from frozen orbitals.
    '''
    moidx = get_frozen_mask(cc)
    nkpts = cc.nkpts
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc

    nao = mo_coeff[0].shape[0]
    dtype = mo_coeff[0].dtype

    nocc_per_kpt = numpy.asarray(get_nocc(cc, per_kpoint=True))
    nmo_per_kpt  = numpy.asarray(get_nmo(cc, per_kpoint=True))

    padded_moidx = []
    for k in range(nkpts):
        kpt_nocc = nocc_per_kpt[k]
        kpt_nvir = nmo_per_kpt[k] - kpt_nocc
        kpt_padded_moidx = numpy.concatenate((numpy.ones(kpt_nocc, dtype=numpy.bool),
                                              numpy.zeros(nmo - kpt_nocc - kpt_nvir, dtype=numpy.bool),
                                              numpy.ones(kpt_nvir, dtype=numpy.bool)))
        padded_moidx.append(kpt_padded_moidx)

    mo_coeff = []
    # Here we will work with two index arrays; one is for our original (small) moidx
    # array while the next is for our new (large) padded array.
    for k in range(nkpts):
        kpt_moidx = moidx[k]
        kpt_padded_moidx = padded_moidx[k]

        mo = numpy.zeros((nao, nmo), dtype=dtype)
        mo[:, kpt_padded_moidx] = cc.mo_coeff[k][:, kpt_moidx]
        mo_coeff.append(mo)

    return mo_coeff

class _ERIS:#(pyscf.cc.ccsd._ChemistsERIs):
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        moidx = get_frozen_mask(cc)
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        #if any(nocc != numpy.count_nonzero(cc._scf.mo_occ[k]>0)
        #       for k in range(nkpts)):
        #    raise NotImplementedError('Different occupancies found for different k-points')

        if mo_coeff is None:
            # If mo_coeff is not canonical orbital
            # TODO does this work for k-points? changed to conjugate.
            raise NotImplementedError
            mo_coeff = cc.mo_coeff
        dtype = mo_coeff[0].dtype

        mo_coeff = pad_frozen_kpt_mo_coeff(cc, mo_coeff)

        # Re-make our fock MO matrix elements from density and fock AO
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc._scf.cell, dm)
        self.fock = numpy.asarray([reduce(numpy.dot,
                                  (mo_coeff[k].T.conj(),fockao[k], mo_coeff[k]))
                                  for k, mo in enumerate(mo_coeff)])

        nocc_per_kpt = numpy.asarray(get_nocc(cc, per_kpoint=True))
        nmo_per_kpt  = numpy.asarray(get_nmo(cc, per_kpoint=True))
        nvir_per_kpt = nmo_per_kpt - nocc_per_kpt
        for kp in range(nkpts):
            mo_e = self.fock[kp].diagonal().real
            gap = abs(mo_e[:nocc_per_kpt[kp]][:, None] -
                      mo_e[-nvir_per_kpt[kp]:]).min()
            if gap < 1e-5:
                logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD at '
                            'k-point %d %s. May cause issues in convergence.',
                            gap, kp, cc.kpts[kp])

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
            self.feri1 = lib.H5TmpFile()

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
                        orbo_p = mo_coeff[kp][:,:nocc]
                        orbo_r = mo_coeff[kr][:,:nocc]
                        buf_kpt = fao2mo((orbo_p,mo_coeff[kq],orbo_r,mo_coeff[ks]),
                                         (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
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
                        orbo_p = mo_coeff[kp][:,:nocc]
                        orbv_r = mo_coeff[kr][:,nocc:]
                        buf_kpt = fao2mo((orbo_p,mo_coeff[kq],orbv_r,mo_coeff[ks]),
                                         (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
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
            #            orbv_p = mo_coeff[kp][:,nocc:]
            #            orbv_q = mo_coeff[kq][:,nocc:]
            #            orbv_r = mo_coeff[kr][:,nocc:]
            #            orbv_s = mo_coeff[ks][:,nocc:]
            #            for a in range(nvir):
            #                orbva_p = orbv_p[:,a].reshape(-1,1)
            #                buf_kpt = fao2mo((orbva_p,orbv_q,orbv_r,orbv_s),
            #                                 (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)
            #                if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
            #                buf_kpt = buf_kpt.reshape((1,nvir,nvir,nvir)).transpose(0,2,1,3)
            #                self.vvvv[kp,kr,kq,a,:,:,:] = buf_kpt[:] / nkpts
            #cput1 = log.timer_debug1('transforming vvvv', *cput1)

            cput1 = time.clock(), time.time()
            for (ikp,ikq,ikr) in khelper.symm_map.keys():
                iks = kconserv[ikp,ikq,ikr]
                orbv_p = mo_coeff[ikp][:,nocc:]
                orbv_q = mo_coeff[ikq][:,nocc:]
                orbv_r = mo_coeff[ikr][:,nocc:]
                orbv_s = mo_coeff[iks][:,nocc:]
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
                        if mo_coeff[0].dtype == np.float: buf_kpt = buf_kpt.real
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
    if maxdiff > 1e-5:
        print("Energy cutoff (or cell.mesh) is not enough to converge AO integrals.")

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
        self._fimd = None

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

        # TODO: check whether to hold Wovov Wovvo in memory
        if self._fimd is None:
            self._fimd = lib.H5TmpFile()
        nkpts, nocc, nvir = t1.shape
        self._fimd.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), t1.dtype.char)
        self._fimd.create_dataset('ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), t1.dtype.char)

        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris,kconserv, self._fimd['ovov'])
        self.Wovvo = imd.Wovvo(t1,t2,eris,kconserv, self._fimd['ovvo'])
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

        nkpts, nocc, nvir = t1.shape
        self._fimd.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), t1.dtype.char)
        self._fimd.create_dataset('ooov', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), t1.dtype.char)
        self._fimd.create_dataset('ovoo', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), t1.dtype.char)

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris,kconserv, self._fimd['oooo'])
        self.Wooov = imd.Wooov(t1,t2,eris,kconserv, self._fimd['ooov'])
        self.Wovoo = imd.Wovoo(t1,t2,eris,kconserv, self._fimd['ovoo'])
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

        nkpts, nocc, nvir = t1.shape
        self._fimd.create_dataset('vovv', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), t1.dtype.char)
        self._fimd.create_dataset('vvvo', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), t1.dtype.char)
        self._fimd.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), t1.dtype.char)

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris,kconserv, self._fimd['vovv'])
        if ea_partition == 'mp' and np.all(t1 == 0):
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,kconserv, self._fimd['vvvo'])
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris,kconserv, self._fimd['vvvv'])
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,kconserv,self.Wvvvv, self._fimd['vvvo'])
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

