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
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.cc import gccsd
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nmo, get_nocc
from pyscf.pbc.cc import kintermediates as imdk
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.pbc.lib import kpts_helper

DEBUG = False

#
# FIXME: When linear dependence is found in KHF and handled by function
# pyscf.scf.addons.remove_linear_dep_, different k-point may have different
# number of orbitals.
#

#einsum = numpy.einsum
einsum = lib.einsum


def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    assert (isinstance(eris, gccsd._PhysicistsERIs))
    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc
        nvir = cc.nmo - nocc
        nkpts = cc.nkpts
        t1 = numpy.zeros((nkpts, nocc, nvir), numpy.complex128)
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nkpts, nocc, nvir = t1.shape
    eold = 0
    eccsd = 0

    if isinstance(cc.diis, lib.diis.DIIS):
        adiis = cc.diis
    elif cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, max_memory)
        normt = numpy.linalg.norm(t1new - t1) + numpy.linalg.norm(t2new - t2)
        if cc.iterative_damping < 1.0:
            alpha = cc.iterative_damping
            t1, t2 = (1-alpha)*t1 + alpha*t1new, (1-alpha)*t2 + alpha*t2new
        else:
            t1, t2 = t1new, t2new
        t1new = t2new = None

        t1, t2 = cc.run_diis(t1, t2, istep, normt, eccsd - eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g', istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd - eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    eris_oovv = eris.oovv.copy()
    e = 0.0 + 0j
    for ki in range(nkpts):
        e += einsum('ia,ia', fock[ki, :nocc, nocc:], t1[ki, :, :])
    t1t1 = numpy.zeros(shape=t2.shape, dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki, kj, ka, :, :, :, :] = einsum('ia,jb->ijab', t1[ki, :, :], t1[kj, :, :])
    tau = t2 + 2 * t1t1
    e += 0.25 * numpy.dot(tau.flatten(), eris_oovv.flatten())
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KCCSD energy %s', e)
    return e.real


def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:, :nocc, nocc:].copy()
    foo = fock[:, :nocc, :nocc].copy()
    fvv = fock[:, nocc:, nocc:].copy()

    tau = imdk.make_tau(cc, t2, t1, t1)

    Fvv = imdk.cc_Fvv(cc, t1, t2, eris)
    Foo = imdk.cc_Foo(cc, t1, t2, eris)
    Fov = imdk.cc_Fov(cc, t1, t2, eris)
    Woooo = imdk.cc_Woooo(cc, t1, t2, eris)
    Wvvvv = imdk.cc_Wvvvv(cc, t1, t2, eris)
    Wovvo = imdk.cc_Wovvo(cc, t1, t2, eris)

    # Move energy terms to the other side
    for k in range(nkpts):
        Fvv[k] -= numpy.diag(numpy.diag(fvv[k]))
        Foo[k] -= numpy.diag(numpy.diag(foo[k]))

    # Get the momentum conservation array
    # Note: chemist's notation for momentum conserving t2(ki,kj,ka,kb), even though
    # integrals are in physics notation
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    eris_ovvo = numpy.zeros(shape=(nkpts, nkpts, nkpts, nocc, nvir, nvir, nocc), dtype=t2.dtype)
    eris_oovo = numpy.zeros(shape=(nkpts, nkpts, nkpts, nocc, nocc, nvir, nocc), dtype=t2.dtype)
    eris_vvvo = numpy.zeros(shape=(nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc), dtype=t2.dtype)
    for km, kb, ke in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ke, kb]
        # <mb||je> -> -<mb||ej>
        eris_ovvo[km, kb, ke] = -eris.ovov[km, kb, kj].transpose(0, 1, 3, 2)
        # <mn||je> -> -<mn||ej>
        # let kb = kn as a dummy variable
        eris_oovo[km, kb, ke] = -eris.ooov[km, kb, kj].transpose(0, 1, 3, 2)
        # <ma||be> -> - <be||am>*
        # let kj = ka as a dummy variable
        kj = kconserv[km, ke, kb]
        eris_vvvo[ke, kj, kb] = -eris.ovvv[km, kb, ke].transpose(2, 3, 1, 0).conj()

    # T1 equation
    t1new = numpy.zeros(shape=t1.shape, dtype=t1.dtype)
    for ka in range(nkpts):
        ki = ka
        t1new[ka] += numpy.array(fov[ka, :, :]).conj()
        t1new[ka] += einsum('ie,ae->ia', t1[ka], Fvv[ka])
        t1new[ka] += -einsum('ma,mi->ia', t1[ka], Foo[ka])
        for km in range(nkpts):
            t1new[ka] += einsum('imae,me->ia', t2[ka, km, ka], Fov[km])
            t1new[ka] += -einsum('nf,naif->ia', t1[km], eris.ovov[km, ka, ki])
            for kn in range(nkpts):
                ke = kconserv[km, ki, kn]
                t1new[ka] += -0.5 * einsum('imef,maef->ia', t2[ki, km, ke], eris.ovvv[km, ka, ke])
                t1new[ka] += -0.5 * einsum('mnae,nmei->ia', t2[km, kn, ka], eris_oovo[kn, km, ke])

    # T2 equation
    t2new = numpy.array(eris.oovv).conj()
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
        kb = kconserv[ki, ka, kj]

        Ftmp = Fvv[kb] - 0.5 * einsum('mb,me->be', t1[kb], Fov[kb])
        tmp = einsum('ijae,be->ijab', t2[ki, kj, ka], Ftmp)
        t2new[ki, kj, ka] += tmp

        #t2new[ki,kj,kb] -= tmp.transpose(0,1,3,2)
        Ftmp = Fvv[ka] - 0.5 * einsum('ma,me->ae', t1[ka], Fov[ka])
        tmp = einsum('ijbe,ae->ijab', t2[ki, kj, kb], Ftmp)
        t2new[ki, kj, ka] -= tmp

        Ftmp = Foo[kj] + 0.5 * einsum('je,me->mj', t1[kj], Fov[kj])
        tmp = einsum('imab,mj->ijab', t2[ki, kj, ka], Ftmp)
        t2new[ki, kj, ka] -= tmp

        #t2new[kj,ki,ka] += tmp.transpose(1,0,2,3)
        Ftmp = Foo[ki] + 0.5 * einsum('ie,me->mi', t1[ki], Fov[ki])
        tmp = einsum('jmab,mi->ijab', t2[kj, ki, ka], Ftmp)
        t2new[ki, kj, ka] += tmp

        for km in range(nkpts):
            # Wminj
            #   - km - kn + ka + kb = 0
            # =>  kn = ka - km + kb
            kn = kconserv[ka, km, kb]
            t2new[ki, kj, ka] += 0.5 * einsum('mnab,mnij->ijab', tau[km, kn, ka], Woooo[km, kn, ki])
            ke = km
            t2new[ki, kj, ka] += 0.5 * einsum('ijef,abef->ijab', tau[ki, kj, ke], Wvvvv[ka, kb, ke])

            # Wmbej
            #     - km - kb + ke + kj = 0
            #  => ke = km - kj + kb
            ke = kconserv[km, kj, kb]
            tmp = einsum('imae,mbej->ijab', t2[ki, km, ka], Wovvo[km, kb, ke])
            #     - km - kb + ke + kj = 0
            # =>  ke = km - kj + kb
            #
            # t[i,e] => ki = ke
            # t[m,a] => km = ka
            if km == ka and ke == ki:
                tmp -= einsum('ie,ma,mbej->ijab', t1[ki], t1[km], eris_ovvo[km, kb, ke])
            t2new[ki, kj, ka] += tmp
            t2new[ki, kj, kb] -= tmp.transpose(0, 1, 3, 2)
            t2new[kj, ki, ka] -= tmp.transpose(1, 0, 2, 3)
            t2new[kj, ki, kb] += tmp.transpose(1, 0, 3, 2)

        ke = ki
        tmp = einsum('ie,abej->ijab', t1[ki], eris_vvvo[ka, kb, ke])
        t2new[ki, kj, ka] += tmp
        # P(ij) term
        ke = kj
        tmp = einsum('je,abei->ijab', t1[kj], eris_vvvo[ka, kb, ke])
        t2new[ki, kj, ka] -= tmp

        km = ka
        tmp = einsum('ma,mbij->ijab', t1[ka], eris.ovoo[km, kb, ki])
        t2new[ki, kj, ka] -= tmp
        # P(ab) term
        km = kb
        tmp = einsum('mb,maij->ijab', t1[kb], eris.ovoo[km, ka, ki])
        t2new[ki, kj, ka] += tmp

    eia = numpy.zeros(shape=(nocc, nvir), dtype=t1new.dtype)
    for ki in range(nkpts):
        eia = foo[ki].diagonal()[:, None] - fvv[ki].diagonal()[None, :]
        # When padding the occupied/virtual arrays, some fock elements will be zero
        idx = numpy.where(abs(eia) < LOOSE_ZERO_TOL)[0]
        eia[idx] = LARGE_DENOM

        t1new[ki] /= eia

    eijab = numpy.zeros(shape=(nocc, nocc, nvir, nvir), dtype=t2new.dtype)
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eijab = (foo[ki].diagonal()[:, None, None, None] + foo[kj].diagonal()[None, :, None, None] -
                 fvv[ka].diagonal()[None, None, :, None] - fvv[kb].diagonal()[None, None, None, :])
        # Due to padding; see above discussion concerning t1new in update_amps()
        idx = numpy.where(abs(eijab) < LOOSE_ZERO_TOL)[0]
        eijab[idx] = LARGE_DENOM

        t2new[ki, kj, ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


class GCCSD(gccsd.GCCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        assert (isinstance(mf, scf.khf.KSCF))
        if not isinstance(mf, scf.kghf.KGHF):
            mf = scf.addons.convert_to_ghf(mf)
        self.kpts = mf.kpts
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    @property
    def nkpts(self):
        return len(self.kpts)

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def dump_flags(self):
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')
        gccsd.GCCSD.dump_flags(self)
        return self

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        t1 = numpy.zeros((nkpts, nocc, nvir), dtype=numpy.complex128)
        t2 = numpy.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=numpy.complex128)
        self.emp2 = 0
        foo = eris.fock[:, :nocc, :nocc].copy()
        fvv = eris.fock[:, nocc:, nocc:].copy()
        fov = eris.fock[:, :nocc, nocc:].copy()
        eris_oovv = eris.oovv.copy()
        eia = numpy.zeros((nocc, nvir))
        eijab = numpy.zeros((nocc, nocc, nvir, nvir))

        kconserv = kpts_helper.get_kconserv(self._scf.cell, self.kpts)
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv[ki, ka, kj]
            eijab = (foo[ki].diagonal()[:, None, None, None] + foo[kj].diagonal()[None, :, None, None] -
                     fvv[ka].diagonal()[None, None, :, None] - fvv[kb].diagonal()[None, None, None, :])
            # Due to padding; see above discussion concerning t1new in update_amps()
            idx = numpy.where(abs(eijab) < LOOSE_ZERO_TOL)[0]
            eijab[idx] = LARGE_DENOM

            t2[ki, kj, ka] = eris_oovv[ki, kj, ka] / eijab

        t2 = numpy.conj(t2)
        self.emp2 = 0.25 * numpy.einsum('pqrijab,pqrijab', t2, eris_oovv).real
        self.emp2 /= nkpts

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2.real)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def ccsd(self, t1=None, t2=None, eris=None, **kwargs):
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       max_memory=self.max_memory, verbose=self.verbose)
        if self.converged:
            logger.info(self, 'CCSD converged')
        else:
            logger.info(self, 'CCSD not converge')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.info(self, 'E(CCSD) = %.16g  E_corr = %.16g', self.e_corr + self._scf.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        nkpts = self.nkpts
        nmo = self.nmo
        mem_incore = nkpts**3 * nmo**4 * 8 / 1e6
        mem_now = lib.current_memory()[0]

        if (mem_incore + mem_now < self.max_memory) or self.mol.incore_anyway:
            return _make_eris_incore(self, mo_coeff)
        else:
            raise NotImplementedError

    def update_amps(self, t1, t2, eris, max_memory=2000):
        return update_amps(self, t1, t2, eris, max_memory)

    def amplitudes_to_vector(self, t1, t2):
        return numpy.hstack((t1.ravel(), t2.ravel()))

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        nov = nkpts * nocc * nvir
        t1 = vec[:nov].reshape(nkpts, nocc, nvir)
        t2 = vec[nov:].reshape(nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)
        return t1, t2


CCSD = GCCSD


def _make_eris_incore(cc, mo_coeff=None):
    log = logger.Logger(cc.stdout, cc.verbose)
    cput0 = (time.clock(), time.time())
    eris = gccsd._PhysicistsERIs()
    kpts = cc.kpts
    nkpts = cc.nkpts
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    eris.nocc = nocc

    #if any(nocc != numpy.count_nonzero(cc._scf.mo_occ[k] > 0) for k in range(nkpts)):
    #    raise NotImplementedError('Different occupancies found for different k-points')

    if mo_coeff is None:
        mo_coeff = cc.mo_coeff
    #else:
    #    # If mo_coeff is not canonical orbital
    #    # TODO does this work for k-points? changed to conjugate.
    #    raise NotImplementedError
    nao = mo_coeff[0].shape[0]
    dtype = mo_coeff[0].dtype

    moidx = get_frozen_mask(cc)
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

    eris.mo_coeff = []
    eris.orbspin = []
    # Generate the molecular orbital coefficients with the frozen orbitals masked.
    # Each MO is tagged with orbspin, a list of 0's and 1's that give the overall
    # spin of each MO.
    #
    # Here we will work with two index arrays; one is for our original (small) moidx
    # array while the next is for our new (large) padded array.
    for k in range(nkpts):
        kpt_moidx = moidx[k]
        kpt_padded_moidx = padded_moidx[k]

        mo = numpy.zeros((nao, nmo), dtype=dtype)
        mo[:, kpt_padded_moidx] = mo_coeff[k][:, kpt_moidx]
        if hasattr(mo_coeff[k], 'orbspin'):
            orbspin_dtype = mo_coeff[k].orbspin[kpt_moidx].dtype
            orbspin = numpy.zeros(nmo, dtype=orbspin_dtype)
            orbspin[kpt_padded_moidx] = mo_coeff[k].orbspin[kpt_moidx]
            mo = lib.tag_array(mo, orbspin=orbspin)
            eris.orbspin.append(orbspin)
        # FIXME: What if the user freezes all up spin orbitals in
        # an RHF calculation?  The number of electrons will still be
        # even.
        else:  # guess orbital spin - assumes an RHF calculation
            assert (numpy.count_nonzero(kpt_moidx) % 2 == 0)
            orbspin = numpy.zeros(mo.shape[1], dtype=int)
            orbspin[1::2] = 1
            mo = lib.tag_array(mo, orbspin=orbspin)
            eris.orbspin.append(orbspin)
        eris.mo_coeff.append(mo)

    # Re-make our fock MO matrix elements from density and fock AO
    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc._scf.cell, dm)
    eris.fock = numpy.asarray([reduce(numpy.dot, (mo.T.conj(), fockao[k], mo)) for k, mo in enumerate(eris.mo_coeff)])

    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    # The bottom nao//2 coefficients are down (up) spin while the top are up (down).
    # These are 'spin-less' quantities; spin-conservation will be added manually.
    if not hasattr(mo_coeff[0], 'orbspin'):
        # The bottom nao//2 coefficients are down (up) spin while the top are up (down).
        mo_a_coeff = [mo[:nao // 2] for mo in eris.mo_coeff]
        mo_b_coeff = [mo[nao // 2:] for mo in eris.mo_coeff]

        eri = numpy.empty((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=numpy.complex128)
        fao2mo = cc._scf.with_df.ao2mo
        for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
            ks = kconserv[kp, kq, kr]
            eri_kpt = fao2mo(
                (mo_a_coeff[kp], mo_a_coeff[kq], mo_a_coeff[kr], mo_a_coeff[ks]), (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                compact=False)
            eri_kpt += fao2mo(
                (mo_b_coeff[kp], mo_b_coeff[kq], mo_b_coeff[kr], mo_b_coeff[ks]), (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                compact=False)
            eri_kpt += fao2mo(
                (mo_a_coeff[kp], mo_a_coeff[kq], mo_b_coeff[kr], mo_b_coeff[ks]), (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                compact=False)
            eri_kpt += fao2mo(
                (mo_b_coeff[kp], mo_b_coeff[kq], mo_a_coeff[kr], mo_a_coeff[ks]), (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                compact=False)

            eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
            eri[kp, kq, kr] = eri_kpt
    else:
        mo_a_coeff = [mo[:nao // 2] + mo[nao // 2:] for mo in eris.mo_coeff]

        eri = numpy.empty((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=numpy.complex128)
        fao2mo = cc._scf.with_df.ao2mo
        for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
            ks = kconserv[kp, kq, kr]
            eri_kpt = fao2mo(
                (mo_a_coeff[kp], mo_a_coeff[kq], mo_a_coeff[kr], mo_a_coeff[ks]), (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                compact=False)

            eri_kpt[(eris.orbspin[kp][:, None] != eris.orbspin[kq]).ravel()] = 0
            eri_kpt[:, (eris.orbspin[kr][:, None] != eris.orbspin[ks]).ravel()] = 0
            eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
            eri[kp, kq, kr] = eri_kpt

    # Check some antisymmetrized properties of the integrals
    if DEBUG:
        check_antisymm_3412(cc, cc.kpts, eri)

    # Antisymmetrizing (pq|rs)-(ps|rq), where the latter integral is equal to
    # (rq|ps); done since we aren't tracking the kpoint of orbital 's'
    eri = eri - eri.transpose(2, 1, 0, 5, 4, 3, 6)
    # Chemist -> physics notation
    eri = eri.transpose(0, 2, 1, 3, 5, 4, 6)

    # Set the various integrals
    eris.dtype = eri.dtype
    eris.oooo = eri[:, :, :, :nocc, :nocc, :nocc, :nocc].copy() / nkpts
    eris.ooov = eri[:, :, :, :nocc, :nocc, :nocc, nocc:].copy() / nkpts
    eris.ovoo = eri[:, :, :, :nocc, nocc:, :nocc, :nocc].copy() / nkpts
    eris.oovv = eri[:, :, :, :nocc, :nocc, nocc:, nocc:].copy() / nkpts
    eris.ovov = eri[:, :, :, :nocc, nocc:, :nocc, nocc:].copy() / nkpts
    eris.ovvv = eri[:, :, :, :nocc, nocc:, nocc:, nocc:].copy() / nkpts
    eris.vvvv = eri[:, :, :, nocc:, nocc:, nocc:, nocc:].copy() / nkpts

    log.timer('CCSD integral transformation', *cput0)
    return eris


def check_antisymm_3412(cc, kpts, integrals):
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    nkpts = len(kpts)
    diff = 0.0
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp, kr, kq]
        for p in range(integrals.shape[3]):
            for q in range(integrals.shape[4]):
                for r in range(integrals.shape[5]):
                    for s in range(integrals.shape[6]):
                        pqrs = integrals[kp, kq, kr, p, q, r, s]
                        rspq = integrals[kq, kp, kr, q, p, r, s]
                        cdiff = numpy.linalg.norm(pqrs - rspq).real
                        if diff > 1e-5:
                            print("AS diff = %.15g" % cdiff, pqrs, rspq, kp, kq, kr, ks, p, q, r, s)
                        diff = max(diff, cdiff)
    print("antisymmetrization : max diff = %.15g" % diff)
    if diff > 1e-5:
        print("Energy cutoff (or cell.mesh) is not enough to converge AO integrals.")
    return diff


def check_antisymm_12(cc, kpts, integrals):
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    nkpts = len(kpts)
    diff = 0.0
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp, kr, kq]
        for p in range(integrals.shape[3]):
            for q in range(integrals.shape[4]):
                for r in range(integrals.shape[5]):
                    for s in range(integrals.shape[6]):
                        pqrs = integrals[kp, kq, kr, p, q, r, s]
                        qprs = integrals[kq, kp, kr, q, p, r, s]
                        cdiff = numpy.linalg.norm(pqrs + qprs).real
                        if diff > 1e-5:
                            print("AS diff = %.15g" % cdiff, pqrs, qprs, kp, kq, kr, ks, p, q, r, s)
                        diff = max(diff, cdiff)
    print("antisymmetrization : max diff = %.15g" % diff)
    if diff > 1e-5:
        print("Energy cutoff (or cell.mesh) is not enough to converge AO integrals.")


def check_antisymm_34(cc, kpts, integrals):
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    nkpts = len(kpts)
    diff = 0.0
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp, kr, kq]
        for p in range(integrals.shape[3]):
            for q in range(integrals.shape[4]):
                for r in range(integrals.shape[5]):
                    for s in range(integrals.shape[6]):
                        pqrs = integrals[kp, kq, kr, p, q, r, s]
                        pqsr = integrals[kp, kq, ks, p, q, s, r]
                        cdiff = numpy.linalg.norm(pqrs + pqsr).real
                        if diff > 1e-5:
                            print("AS diff = %.15g" % cdiff, pqrs, pqsr, kp, kq, kr, ks, p, q, r, s)
                        diff = max(diff, cdiff)
    print("antisymmetrization : max diff = %.15g" % diff)
    if diff > 1e-5:
        print("Energy cutoff (or cell.mesh) is not enough to converge AO integrals.")
