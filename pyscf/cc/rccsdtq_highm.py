#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Yu Jin <yjin@flatironinstitute.org>
#         Huanchen Zhai <hczhai.ok@gmail.com>
#

'''
RHF-CCSDTQ with full T4 amplitudes stored.
T1-dressed formalism is used, where the T1 amplitudes are absorbed into the Fock matrix and ERIs.

Ref:
J. Chem. Phys. 142, 064108 (2015); DOI:10.1063/1.4907278
Chem. Phys. Lett. 228, 233 (1994); DOI:10.1016/0009-2614(94)00898-1
'''

import numpy as np
import numpy
import functools
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf.cc import _ccsd, rccsdtq
from pyscf.cc.rccsdt import (_einsum, t3_spin_summation_inplace_, update_t1_fock_eris, intermediates_t1t2,
                            compute_r1r2, r1r2_divide_e_, intermediates_t3, _PhysicistsERIs)
from pyscf.cc.rccsdt_highm import (t3_spin_summation, t3_perm_symmetrize_inplace_, purify_tamps_, r1r2_add_t3_,
                                    intermediates_t3_add_t3, compute_r3, r3_divide_e_)
from pyscf.cc.rccsdtq import t4_spin_summation_inplace_, t4_add_, _IMDS
from pyscf import __config__


_libccsdt = lib.load_library('libccsdt')

def t4_spin_summation(A, B, nocc4, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    assert B.dtype == np.float64 and B.flags['C_CONTIGUOUS'], "B must be a contiguous float64 array"
    pattern_c = pattern.encode('utf-8')
    _libccsdt.t4_spin_summation(
        A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4), ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return B

def t4_perm_symmetrize_inplace_(A, nocc, nvir, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    _libccsdt.t4_perm_symmetrize_inplace_(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return A

def eijkl_division_(A, eia, nocc, nvir):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    assert eia.dtype == np.float64 and eia.flags['C_CONTIGUOUS'], "eia must be a contiguous float64 array"
    _libccsdt.eijkl_division_(
        A.ctypes.data_as(ctypes.c_void_p), eia.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
    )
    return A

def r2_add_t4_(mycc, imds, r2, t4):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    t1_eris = imds.t1_eris

    c_t4 = np.empty_like(t4)
    t4_spin_summation(t4, c_t4, nocc**4, nvir, "P4_442", 1.0, 0.0)
    einsum('mnef,mnijefab->ijab', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t4, out=r2, alpha=0.25, beta=1.0)
    c_t4 = None
    return r2

def r3_add_t4_(mycc, imds, r3, t4):
    '''Add the T4 contributions to r3. T4 amplitudes are stored in full form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    c_t4 = np.empty_like(t4)
    t4_spin_summation(t4, c_t4, nocc**4, nvir, "P4_201", 1.0, 0.0)
    einsum('me,mijkeabc->ijkabc', t1_fock[:nocc, nocc:], c_t4, out=r3, alpha=1.0 / 6.0, beta=1.0)
    einsum('amef,mijkfebc->ijkabc', t1_eris[nocc:, :nocc, nocc:, nocc:], c_t4, out=r3, alpha=0.5, beta=1.0)
    einsum('mnej,minkeabc->ijkabc', t1_eris[:nocc, :nocc, nocc:, :nocc], c_t4, out=r3, alpha=-0.5, beta=1.0)
    c_t4 = None
    return r3

def intermediates_t4(mycc, imds, t2, t3, t4):
    '''Intermediates for the T4 residual equation, with T4 amplitudes stored in full form.
    In place modification of W_vvvo.
    '''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    W_vvvo, W_oooo, W_ovov, W_vvvv = imds.W_vvvo, imds.W_oooo, imds.W_ovov, imds.W_vvvv

    einsum('me,mjab->abej', t1_fock[:nocc, nocc:], t2, out=W_vvvo, alpha=-1.0, beta=1.0)

    W_ovvvoo = np.empty((nocc,) + (nvir,) * 3 + (nocc,) * 2, dtype=t2.dtype)
    einsum('maef,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=2.0, beta=0.0)
    einsum('mafe,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=-1.0, beta=1.0)
    einsum('mnei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=-2.0, beta=1.0)
    einsum('nmei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=1.0, beta=1.0)
    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)
    einsum('nmfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=0.5, beta=1.0)
    einsum('mnfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=-0.25, beta=1.0)
    c_t3 = None

    W_ovvovo = np.empty((nocc,) + (nvir,) * 2 + (nocc, nvir, nocc), dtype=t2.dtype)
    einsum('mafe,jibf->mabiej', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvovo, alpha=1.0, beta=0.0)
    einsum('mnie,njab->mabiej', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_ovvovo, alpha=-1.0, beta=1.0)
    einsum('nmef,injfab->mabiej', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ovvovo, alpha=-0.5, beta=1.0)

    W_vooooo = np.empty((nvir,) + (nocc,) * 5, dtype=t2.dtype)
    einsum('mnek,ijae->amnijk', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vooooo, alpha=1.0, beta=0.0)
    einsum('mnef,ijkaef->amnijk', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_vooooo, alpha=0.5, beta=1.0)
    W_vooooo += W_vooooo.transpose(0, 2, 1, 3, 5, 4)

    W_vvoooo = np.empty((nvir,) * 2 + (nocc,) * 4, dtype=t2.dtype)
    einsum('amef,ijkebf->abmijk', t1_eris[nocc:, :nocc, nocc:, nocc:], t3, out=W_vvoooo, alpha=1.0, beta=0.0)
    tmp_ovvo = t1_eris[:nocc, nocc:, nocc:, :nocc].copy()
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    einsum('nmfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=tmp_ovvo, alpha=1.0, beta=1.0)
    einsum('mnfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=tmp_ovvo, alpha=-0.5, beta=1.0)
    c_t2 = None
    einsum('nmef,infa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=tmp_ovvo, alpha=-0.5, beta=1.0)
    einsum('maei,jkbe->abmijk', tmp_ovvo, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    tmp_ovvo = None
    einsum('make,jibe->abmijk', W_ovov, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum('mnki,njab->abmijk', W_oooo, t2, out=W_vvoooo, alpha=-0.5, beta=1.0)
    c_t4 = np.empty_like(t4)
    t4_spin_summation(t4, c_t4, nocc**4, nvir, "P4_201", 1.0, 0.0)
    einsum('mnef,nijkfabe->abmijk', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t4, out=W_vvoooo, alpha=0.5, beta=1.0)

    W_vvvvoo = np.empty((nvir,) * 4 + (nocc,) * 2, dtype=t2.dtype)
    einsum('abef,jkfc->abcejk', W_vvvv, t2, out=W_vvvvoo, alpha=0.5, beta=0.0)
    einsum('mnef,nmjkfabc->abcejk', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t4, out=W_vvvvoo, alpha=-0.5, beta=1.0)
    c_t4 = None

    W_ovvvoo += W_ovvvoo.transpose(0, 2, 1, 3, 5, 4)
    W_vvoooo += W_vvoooo.transpose(1, 0, 2, 4, 3, 5)
    W_vvvvoo += W_vvvvoo.transpose(0, 2, 1, 3, 5, 4)
    imds.W_ovvvoo, imds.W_ovvovo, imds.W_vooooo = W_ovvvoo, W_ovvovo, W_vooooo
    imds.W_vvoooo, imds.W_vvvvoo = W_vvoooo, W_vvvvoo
    return imds

def compute_r4(mycc, imds, t2, t3, t4):
    '''Compute r4 with full T4 amplitudes; r4 is returned in full form as well.
    r4 will require a symmetry restoration step afterward.
    '''
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    F_oo, F_vv = imds.F_oo, imds.F_vv
    W_oooo, W_ovvo, W_ovov = imds.W_oooo, imds.W_ovvo, imds.W_ovov
    W_vvvo, W_vooo, W_vvvv = imds.W_vvvo, imds.W_vooo, imds.W_vvvv
    W_ovvvoo, W_ovvovo, W_vooooo = imds.W_ovvvoo, imds.W_ovvovo, imds.W_vooooo
    W_vvoooo, W_vvvvoo = imds.W_vvoooo, imds.W_vvvvoo

    r4 = np.empty_like(t4)
    einsum('abej,iklecd->ijklabcd', W_vvvo, t3, out=r4, alpha=0.5, beta=0.0)
    W_vvvo = imds.W_vvvo = None
    time1 = log.timer_debug1('t4: W_vvvo * t3', *time1)

    einsum('amij,mklbcd->ijklabcd', W_vooo, t3, out=r4, alpha=-0.5, beta=1.0)
    W_vooo = imds.W_vooo = None
    time1 = log.timer_debug1('t4: W_vooo * t3', *time1)

    einsum('ae,ijklebcd->ijklabcd', F_vv, t4, out=r4, alpha=1.0 / 6.0, beta=1.0)
    F_vv = imds.F_vv = None
    time1 = log.timer_debug1('t4: F_vv * t4', *time1)

    einsum('mi,mjklabcd->ijklabcd', F_oo, t4, out=r4, alpha=-1.0 / 6.0, beta=1.0)
    F_oo = imds.F_oo = None
    time1 = log.timer_debug1('t4: F_oo * t4', *time1)

    c_t4 = np.empty_like(t4)
    t4_spin_summation(t4, c_t4, nocc**4, nvir, "P4_201", 1.0, 0.0)
    einsum('maei,mjklebcd->ijklabcd', W_ovvo, c_t4, out=r4, alpha=1.0 / 12.0, beta=1.0)
    c_t4 = None
    W_ovvo = imds.W_ovvo = None
    time1 = log.timer_debug1('t4: W_ovvo * c_t4', *time1)

    einsum('maie,jmklebcd->ijklabcd', W_ovov, t4, out=r4, alpha=-0.25, beta=1.0)
    einsum('mbie,jmkleacd->ijklabcd', W_ovov, t4, out=r4, alpha=-0.5, beta=1.0)
    W_ovov = imds.W_ovov = None
    time1 = log.timer_debug1('t4: W_ovov * t4', *time1)

    einsum('mnij,mnklabcd->ijklabcd', W_oooo, t4, out=r4, alpha=0.25, beta=1.0)
    W_oooo = imds.W_oooo = None
    time1 = log.timer_debug1('t4: W_oooo * t4', *time1)

    einsum('abef,ijklefcd->ijklabcd', W_vvvv, t4, out=r4, alpha=0.25, beta=1.0)
    W_vvvv = imds.W_vvvv = None
    time1 = log.timer_debug1('t4: W_vvvv * t4', *time1)

    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)
    einsum('mabeij,mklecd->ijklabcd', W_ovvvoo, c_t3, out=r4, alpha=0.125, beta=1.0)
    W_ovvvoo = imds.W_ovvvoo = None
    c_t3 = None
    time1 = log.timer_debug1('t4: W_ovvvoo * c_t3', *time1)

    einsum('mabiej,kmlecd->ijklabcd', W_ovvovo, t3, out=r4, alpha=-0.5, beta=1.0)
    einsum('mcbiej,kmlead->ijklabcd', W_ovvovo, t3, out=r4, alpha=-1.0, beta=1.0)
    W_ovvovo = imds.W_ovvovo = None
    time1 = log.timer_debug1('t4: W_ovvovo * t3', *time1)

    einsum('amnijk,mnlbcd->ijklabcd', W_vooooo, t3, out=r4, alpha=0.5, beta=1.0)
    W_vooooo = imds.W_vooooo = None
    time1 = log.timer_debug1('t4: W_vooooo * t3', *time1)

    einsum('abmijk,mlcd->ijklabcd', W_vvoooo, t2, out=r4, alpha=-0.5, beta=1.0)
    W_vvoooo = imds.W_vvoooo = None
    time1 = log.timer_debug1('t4: W_vvoooo * t2', *time1)

    einsum('abcejk,iled->ijklabcd', W_vvvvoo, t2, out=r4, alpha=0.5, beta=1.0)
    W_vvvvoo = imds.W_vvvvoo = None
    time1 = log.timer_debug1('t4: W_vvvvoo * t2', *time1)
    return r4

def r4_divide_e_(mycc, r4, mo_energy):
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:] - mycc.level_shift
    eijkl_division_(r4, eia, nocc, nvir)
    return r4

def update_amps_rccsdtq_(mycc, tamps, eris):
    '''Update RCCSDTQ amplitudes in place, with T4 amplitudes stored in full form.'''
    assert (isinstance(eris, _PhysicistsERIs))
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1, t2, t3, t4 = tamps
    mo_energy = eris.mo_energy

    imds = _IMDS

    # t1, t2
    update_t1_fock_eris(mycc, imds, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2(mycc, imds, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, imds, t2)
    r1r2_add_t3_(mycc, imds, r1, r2, t3)
    r2_add_t4_(mycc, imds, r2, t4)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # symmetrization
    r2 += r2.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t1t2: symmetrize r2', *time1)
    # divide by eijkabc
    r1r2_divide_e_(mycc, r1, r2, mo_energy)
    time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)

    res_norm = [np.linalg.norm(r1), np.linalg.norm(r2)]

    t1 += r1
    t2 += r2
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)

    # t3
    intermediates_t3(mycc, imds, t2)
    intermediates_t3_add_t3(mycc, imds, t3)
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3(mycc, imds, t2, t3)
    r3_add_t4_(mycc, imds, r3, t4)
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # symmetrization
    t3_perm_symmetrize_inplace_(r3, nocc, nvir, 1.0, 0.0)
    t3_spin_summation_inplace_(r3, nocc**3, nvir, "P3_full", -1.0 / 6.0, 1.0)
    purify_tamps_(r3)
    time1 = log.timer_debug1('t3: symmetrize r3', *time1)
    # divide by eijkabc
    r3_divide_e_(mycc, r3, mo_energy)
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm.append(np.linalg.norm(r3))

    t3 += r3
    r3 = None
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)

    # t4
    intermediates_t4(mycc, imds, t2, t3, t4)
    imds.t1_fock, imds.t1_eris = None, None
    time1 = log.timer_debug1('t4: update intermediates', *time0)
    r4 = compute_r4(mycc, imds, t2, t3, t4)
    imds = None
    time1 = log.timer_debug1('t4: compute r4', *time1)
    # symmetrization
    t4_perm_symmetrize_inplace_(r4, nocc, nvir, 1.0, 0.0)
    t4_spin_summation_inplace_(r4, nocc**4, nvir, "P4_full", -1.0 / 24.0, 1.0)
    purify_tamps_(r4)
    time1 = log.timer_debug1('t4: symmetrize r4', *time1)
    # divide by eijkabc
    r4_divide_e_(mycc, r4, mo_energy)
    time1 = log.timer_debug1('t4: divide r4 by eijklabcd', *time1)

    res_norm.append(np.linalg.norm(r4))

    t4_add_(t4, r4, nocc**4, nvir)
    r4 = None
    time1 = log.timer_debug1('t4: update t4', *time1)
    time0 = log.timer_debug1('t4 total', *time0)
    return res_norm


class RCCSDTQ(rccsdtq.RCCSDTQ):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        rccsdtq.RCCSDTQ.__init__(self, mf, frozen, mo_coeff, mo_occ)

    do_tri_max_t = property(lambda self: False)

    update_amps_ = update_amps_rccsdtq_


if __name__ == "__main__":

    from pyscf import gto, scf

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", verbose=3)
    mf = scf.RHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()
    print()
    ref_ecorr = -0.157579406507473
    frozen = 0
    mycc = RCCSDTQ(mf, frozen=frozen)
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.do_diis_max_t = True
    mycc.incore_complete = True
    mycc.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_ecorr, mycc.e_corr - ref_ecorr))
    print()
