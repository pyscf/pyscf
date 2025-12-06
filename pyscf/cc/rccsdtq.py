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
RHF-CCSDTQ with T4 amplitudes stored only for the i <= j <= k <= l index combinations.
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
from pyscf.cc import ccsd, _ccsd, rccsdt
from pyscf.cc.rccsdt import (_einsum, t3_spin_summation_inplace_, symmetrize_tamps_tri_, purify_tamps_tri_,
                            update_t1_fock_eris, intermediates_t1t2, compute_r1r2, r1r2_divide_e_,
                            intermediates_t3, kernel, _PhysicistsERIs, format_size)
from pyscf.cc.rccsdt_highm import (t3_spin_summation, t3_perm_symmetrize_inplace_, purify_tamps_, r1r2_add_t3_,
                                    intermediates_t3_add_t3, compute_r3, r3_divide_e_)
from pyscf import __config__


_libccsdt = lib.load_library('libccsdt')

def t4_spin_summation_inplace_(A, nocc4, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    pattern_c = pattern.encode('utf-8')
    _libccsdt.t4_spin_summation_inplace_(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4), ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return A

def t4_add_(t4, r4, nocc4, nvir):
    assert t4.dtype == np.float64 and t4.flags['C_CONTIGUOUS'], "t4 must be a contiguous float64 array"
    assert r4.dtype == np.float64 and r4.flags['C_CONTIGUOUS'], "r4 must be a contiguous float64 array"
    _libccsdt.t4_add_(
        t4.ctypes.data_as(ctypes.c_void_p), r4.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc4), ctypes.c_int64(nvir),
    )
    return t4

def unpack_t4_tri2block_(t4, t4_blk, map_, mask, i0, i1, j0, j1, k0, k1, l0, l1,
                            nocc, nvir, blk_i, blk_j, blk_k, blk_l):
    assert t4.dtype == np.float64 and t4_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_
    t4 = np.ascontiguousarray(t4)
    t4_blk = np.ascontiguousarray(t4_blk)
    map_ = np.ascontiguousarray(map_)
    mask = np.ascontiguousarray(mask)
    _libccsdt.unpack_t4_tri2block_(
        t4.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t4_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(l0), ctypes.c_int64(l1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k), ctypes.c_int64(blk_l)
    )
    return t4_blk

def accumulate_t4_block2tri_(t4, t4_blk, map_, i0, i1, j0, j1, k0, k1, l0, l1,
                                nocc, nvir, blk_i, blk_j, blk_k, blk_l, alpha, beta):
    assert t4.dtype == np.float64 and t4_blk.dtype == np.float64
    assert map_.dtype == np.int64
    t4 = np.ascontiguousarray(t4)
    t4_blk = np.ascontiguousarray(t4_blk)
    map_ = np.ascontiguousarray(map_)
    _libccsdt.accumulate_t4_block2tri_(
        t4.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t4_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(l0), ctypes.c_int64(l1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k), ctypes.c_int64(blk_l),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t4

def _unpack_t4_(mycc, t4, t4_blk, i0, i1, j0, j1, k0, k1, l0, l1,
                    blksize0=None, blksize1=None, blksize2=None, blksize3=None):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    if blksize3 is None: blksize3 = mycc.blksize
    unpack_t4_tri2block_(t4, t4_blk, mycc.tri2block_map, mycc.tri2block_mask, i0, i1, j0, j1, k0, k1, l0, l1,
                        mycc.nocc, mycc.nmo - mycc.nocc, blksize0, blksize1, blksize2, blksize3)
    return t4_blk

def _accumulate_t4_(mycc, t4, t4_blk, i0, i1, j0, j1, k0, k1, l0, l1,
                    blksize0=None, blksize1=None, blksize2=None, blksize3=None, alpha=1.0, beta=0.0):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    if blksize3 is None: blksize3 = mycc.blksize
    accumulate_t4_block2tri_(t4, t4_blk, mycc.tri2block_map, i0, i1, j0, j1, k0, k1, l0, l1, mycc.nocc,
                                mycc.nmo - mycc.nocc, blksize0, blksize1, blksize2, blksize3, alpha=alpha, beta=beta)
    return t4

def r2_add_t4_tri_(mycc, imds, r2, t4):
    '''Add the T4 contributions to r2. T4 amplitudes are stored in triangular form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    t1_eris = imds.t1_eris

    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=np.float64)
    for m0, m1 in lib.prange(0, nocc, blksize):
        bm = m1 - m0
        for n0, n1 in lib.prange(0, nocc, blksize):
            bn = n1 - n0
            for i0, i1 in lib.prange(0, nocc, blksize):
                bi = i1 - i0
                for j0, j1 in lib.prange(0, nocc, blksize):
                    bj = j1 - j0
                    _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, j0, j1)
                    t4_spin_summation_inplace_(t4_tmp, blksize**4, nvir, "P4_442", 1.0, 0.0)
                    einsum('mnef,mnijefab->ijab', t1_eris[m0:m1, n0:n1, nocc:, nocc:],
                        t4_tmp[:bm, :bn, :bi, :bj], out=r2[i0:i1, j0:j1, :, :], alpha=0.25, beta=1.0)
    t4_tmp = None
    return r2

def r3_add_t4_tri_(mycc, imds, r3, t4):
    '''Add the T4 contributions to r3. T4 amplitudes are stored in triangular form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for m0, m1 in lib.prange(0, nocc, blksize):
        bm = m1 - m0
        for i0, i1 in lib.prange(0, nocc, blksize):
            bi = i1 - i0
            for j0, j1 in lib.prange(0, nocc, blksize):
                bj = j1 - j0
                for k0, k1 in lib.prange(0, nocc, blksize):
                    bk = k1 - k0
                    _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                    t4_spin_summation_inplace_(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                    einsum('me,mijkeabc->ijkabc', t1_fock[m0:m1, nocc:], t4_tmp[:bm, :bi, :bj, :bk],
                            out=r3[i0:i1, j0:j1, k0:k1, ...], alpha=1.0 / 6.0, beta=1.0)
                    einsum('amef,mijkfebc->ijkabc', t1_eris[nocc:, m0:m1, nocc:, nocc:],
                        t4_tmp[:bm, :bi, :bj, :bk], out=r3[i0:i1, j0:j1, k0:k1, ...], alpha=0.5, beta=1.0)
                    einsum('mjen,mijkeabc->inkabc', t1_eris[m0:m1, j0:j1, nocc:, :nocc],
                        t4_tmp[:bm, :bi, :bj, :bk], out=r3[i0:i1, :, k0:k1, ...], alpha=-0.5, beta=1.0)
    t4_tmp = None
    return r3

def intermediates_t4_tri(mycc, imds, t2, t3, t4):
    '''Intermediates for the T4 residual equation, with T4 amplitudes stored in triangular form.
    In place modification of W_vvvo.
    '''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    W_vvvo, W_oooo, W_ovov, W_vvvv = imds.W_vvvo, imds.W_oooo, imds.W_ovov, imds.W_vvvv

    einsum('me,mjab->abej', t1_fock[:nocc, nocc:], t2, out=W_vvvo, alpha=-1.0, beta=1.0)

    W_ovvvoo = np.empty((nocc,) + (nvir,) * 3 + (nocc,) * 2)
    einsum('maef,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=2.0, beta=0.0)
    einsum('mafe,jibf->mabeij', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvvoo, alpha=-1.0, beta=1.0)
    einsum('mnei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=-2.0, beta=1.0)
    einsum('nmei,njab->mabeij', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ovvvoo, alpha=1.0, beta=1.0)
    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)
    einsum('nmfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=0.5, beta=1.0)
    einsum('mnfe,nijfab->mabeij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_ovvvoo, alpha=-0.25, beta=1.0)
    c_t3 = None

    W_ovvovo = np.empty((nocc, nvir, nvir, nocc, nvir, nocc))
    einsum('mafe,jibf->mabiej', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovvovo, alpha=1.0, beta=0.0)
    einsum('mnie,njab->mabiej', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_ovvovo, alpha=-1.0, beta=1.0)
    einsum('nmef,injfab->mabiej', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ovvovo, alpha=-0.5, beta=1.0)

    W_vooooo = np.empty((nvir,) + (nocc,) * 5)
    einsum('mnek,ijae->amnijk', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vooooo, alpha=1.0, beta=0.0)
    einsum('mnef,ijkaef->amnijk', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_vooooo, alpha=0.5, beta=1.0)
    W_vooooo += W_vooooo.transpose(0, 2, 1, 3, 5, 4)

    W_vvoooo = np.empty((nvir,) * 2 + (nocc,) * 4)
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

    W_vvvvoo = np.empty((nvir,) * 4 + (nocc,) * 2)
    einsum('abef,jkfc->abcejk', W_vvvv, t2, out=W_vvvvoo, alpha=0.5, beta=0.0)

    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for n0, n1 in lib.prange(0, nocc, blksize):
        bn = n1 - n0
        for i0, i1 in lib.prange(0, nocc, blksize):
            bi = i1 - i0
            for j0, j1 in lib.prange(0, nocc, blksize):
                bj = j1 - j0
                for k0, k1 in lib.prange(0, nocc, blksize):
                    bk = k1 - k0
                    _unpack_t4_(mycc, t4, t4_tmp, n0, n1, i0, i1, j0, j1, k0, k1)
                    t4_spin_summation_inplace_(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                    einsum('mnef,nijkfabe->abmijk', t1_eris[:nocc, n0:n1, nocc:, nocc:],
                        t4_tmp[:bn, :bi, :bj, :bk], out=W_vvoooo[..., i0:i1, j0:j1, k0:k1], alpha=0.5, beta=1.0)
                    einsum('inef,nijkfabc->abcejk', t1_eris[i0:i1, n0:n1, nocc:, nocc:],
                        t4_tmp[:bn, :bi, :bj, :bk], out=W_vvvvoo[..., j0:j1, k0:k1], alpha=-0.5, beta=1.0)
    t4_tmp = None

    W_ovvvoo += W_ovvvoo.transpose(0, 2, 1, 3, 5, 4)
    W_vvoooo += W_vvoooo.transpose(1, 0, 2, 4, 3, 5)
    W_vvvvoo += W_vvvvoo.transpose(0, 2, 1, 3, 5, 4)
    imds.W_ovvvoo, imds.W_ovvovo, imds.W_vooooo = W_ovvvoo, W_ovvovo, W_vooooo
    imds.W_vvoooo, imds.W_vvvvoo = W_vvoooo, W_vvvvoo
    return imds

def compute_r4_tri(mycc, imds, t2, t3, t4):
    '''Compute r4 with triangular-stored T4 amplitudes; r4 is returned in triangular form as well.
    r4 will require a symmetry restoration step afterward.
    '''
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo, blksize = mycc.nocc, mycc.nmo, mycc.blksize
    nvir = nmo - nocc

    F_oo, F_vv = imds.F_oo, imds.F_vv
    W_oooo, W_ovvo, W_ovov = imds.W_oooo, imds.W_ovvo, imds.W_ovov
    W_vvvo, W_vooo, W_vvvv = imds.W_vvvo, imds.W_vooo, imds.W_vvvv
    W_ovvvoo, W_ovvovo, W_vooooo = imds.W_ovvvoo, imds.W_ovvovo, imds.W_vooooo
    W_vvoooo, W_vvvvoo = imds.W_vvoooo, imds.W_vvvvoo

    c_t3 = np.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)

    # r4 = np.empty_like(t4)
    r4 = np.zeros_like(t4)

    time2 = logger.process_clock(), logger.perf_counter()
    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0

                    einsum("abej,iklecd->ijklabcd", W_vvvo[..., j0:j1], t3[i0:i1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=0.0)
                    einsum("acek,ijlebd->ijklabcd", W_vvvo[..., k0:k1], t3[i0:i1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("adel,ijkebc->ijklabcd", W_vvvo[..., l0:l1], t3[i0:i1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("baei,jklecd->ijklabcd", W_vvvo[..., i0:i1], t3[j0:j1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("caei,kjlebd->ijklabcd", W_vvvo[..., i0:i1], t3[k0:k1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("daei,ljkebc->ijklabcd", W_vvvo[..., i0:i1], t3[l0:l1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("bcek,jilead->ijklabcd", W_vvvo[..., k0:k1], t3[j0:j1, i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("bdel,jikeac->ijklabcd", W_vvvo[..., l0:l1], t3[j0:j1, i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("cbej,kilead->ijklabcd", W_vvvo[..., j0:j1], t3[k0:k1, i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("dbej,likeac->ijklabcd", W_vvvo[..., j0:j1], t3[l0:l1, i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("cdel,kijeab->ijklabcd", W_vvvo[..., l0:l1], t3[k0:k1, i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("dcek,lijeab->ijklabcd", W_vvvo[..., k0:k1], t3[l0:l1, i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    einsum("amij,mklbcd->ijklabcd", W_vooo[:, :, i0:i1, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("amik,mjlcbd->ijklabcd", W_vooo[:, :, i0:i1, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("amil,mjkdbc->ijklabcd", W_vooo[:, :, i0:i1, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("bmji,mklacd->ijklabcd", W_vooo[:, :, j0:j1, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("cmki,mjlabd->ijklabcd", W_vooo[:, :, k0:k1, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("dmli,mjkabc->ijklabcd", W_vooo[:, :, l0:l1, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("bmjk,milcad->ijklabcd", W_vooo[:, :, j0:j1, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("bmjl,mikdac->ijklabcd", W_vooo[:, :, j0:j1, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("cmkj,milbad->ijklabcd", W_vooo[:, :, k0:k1, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("dmlj,mikbac->ijklabcd", W_vooo[:, :, l0:l1, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("cmkl,mijdab->ijklabcd", W_vooo[:, :, k0:k1, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("dmlk,mijcab->ijklabcd", W_vooo[:, :, l0:l1, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    einsum("mabeij,mklecd->ijklabcd", W_ovvvoo[..., i0:i1, j0:j1],
                        c_t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("maceik,mjlebd->ijklabcd", W_ovvvoo[..., i0:i1, k0:k1],
                        c_t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("madeil,mjkebc->ijklabcd", W_ovvvoo[..., i0:i1, l0:l1],
                        c_t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mbaeji,mklecd->ijklabcd", W_ovvvoo[..., j0:j1, i0:i1],
                        c_t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mcaeki,mjlebd->ijklabcd", W_ovvvoo[..., k0:k1, i0:i1],
                        c_t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mdaeli,mjkebc->ijklabcd", W_ovvvoo[..., l0:l1, i0:i1],
                        c_t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mbcejk,milead->ijklabcd", W_ovvvoo[..., j0:j1, k0:k1],
                        c_t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mbdejl,mikeac->ijklabcd", W_ovvvoo[..., j0:j1, l0:l1],
                        c_t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mcbekj,milead->ijklabcd", W_ovvvoo[..., k0:k1, j0:j1],
                        c_t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mdbelj,mikeac->ijklabcd", W_ovvvoo[..., l0:l1, j0:j1],
                        c_t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mcdekl,mijeab->ijklabcd", W_ovvvoo[..., k0:k1, l0:l1],
                        c_t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)
                    einsum("mdcelk,mijeab->ijklabcd", W_ovvvoo[..., l0:l1, k0:k1],
                        c_t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.25, beta=1.0)

                    einsum("mabiej,mklced->ijklabcd", W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mabiej,mlkdec->ijklabcd", W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("maciek,mjlbed->ijklabcd", W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("madiel,mjkbec->ijklabcd", W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("maciek,mljdeb->ijklabcd", W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("madiel,mkjceb->ijklabcd", W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mbajei,mklced->ijklabcd", W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mbajei,mlkdec->ijklabcd", W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mcakei,mjlbed->ijklabcd", W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mdalei,mjkbec->ijklabcd", W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mcakei,mljdeb->ijklabcd", W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mdalei,mkjceb->ijklabcd", W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mbcjek,milaed->ijklabcd", W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mbdjel,mikaec->ijklabcd", W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mcbkej,milaed->ijklabcd", W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mdblej,mikaec->ijklabcd", W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mcdkel,mijaeb->ijklabcd", W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mdclek,mijaeb->ijklabcd", W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mbcjek,mlidea->ijklabcd", W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mbdjel,mkicea->ijklabcd", W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mcbkej,mlidea->ijklabcd", W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mdblej,mkicea->ijklabcd", W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mcdkel,mjibea->ijklabcd", W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    einsum("mdclek,mjibea->ijklabcd", W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)

                    einsum("mcbiej,mklaed->ijklabcd", W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mdbiej,mlkaec->ijklabcd", W_ovvovo[..., i0:i1, :, j0:j1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mbciek,mjlaed->ijklabcd", W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mbdiel,mjkaec->ijklabcd", W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mdciek,mljaeb->ijklabcd", W_ovvovo[..., i0:i1, :, k0:k1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mcdiel,mkjaeb->ijklabcd", W_ovvovo[..., i0:i1, :, l0:l1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mcajei,mklbed->ijklabcd", W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, k0:k1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mdajei,mlkbec->ijklabcd", W_ovvovo[..., j0:j1, :, i0:i1],
                        t3[:, l0:l1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mbakei,mjlced->ijklabcd", W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, j0:j1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mbalei,mjkdec->ijklabcd", W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, j0:j1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mdakei,mljceb->ijklabcd", W_ovvovo[..., k0:k1, :, i0:i1],
                        t3[:, l0:l1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mcalei,mkjdeb->ijklabcd", W_ovvovo[..., l0:l1, :, i0:i1],
                        t3[:, k0:k1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("macjek,milbed->ijklabcd", W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("madjel,mikbec->ijklabcd", W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mabkej,milced->ijklabcd", W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, i0:i1, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mablej,mikdec->ijklabcd", W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, i0:i1, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("madkel,mijceb->ijklabcd", W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("maclek,mijdeb->ijklabcd", W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, i0:i1, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mdcjek,mlibea->ijklabcd", W_ovvovo[..., j0:j1, :, k0:k1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mcdjel,mkibea->ijklabcd", W_ovvovo[..., j0:j1, :, l0:l1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mdbkej,mlicea->ijklabcd", W_ovvovo[..., k0:k1, :, j0:j1],
                        t3[:, l0:l1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mcblej,mkidea->ijklabcd", W_ovvovo[..., l0:l1, :, j0:j1],
                        t3[:, k0:k1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mbdkel,mjicea->ijklabcd", W_ovvovo[..., k0:k1, :, l0:l1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mbclek,mjidea->ijklabcd", W_ovvovo[..., l0:l1, :, k0:k1],
                        t3[:, j0:j1, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    einsum("amnijk,mnlbcd->ijklabcd", W_vooooo[..., i0:i1, j0:j1, k0:k1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("amnijl,mnkbdc->ijklabcd", W_vooooo[..., i0:i1, j0:j1, l0:l1],
                        t3[:, :, k0:k1,], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("amnikl,mnjcdb->ijklabcd", W_vooooo[..., i0:i1, k0:k1, l0:l1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("bmnjik,mnlacd->ijklabcd", W_vooooo[..., j0:j1, i0:i1, k0:k1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("bmnjil,mnkadc->ijklabcd", W_vooooo[..., j0:j1, i0:i1, l0:l1],
                        t3[:, :, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("bmnjkl,mnicda->ijklabcd", W_vooooo[..., j0:j1, k0:k1, l0:l1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("cmnkij,mnlabd->ijklabcd", W_vooooo[..., k0:k1, i0:i1, j0:j1],
                        t3[:, :, l0:l1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("cmnkil,mnjadb->ijklabcd", W_vooooo[..., k0:k1, i0:i1, l0:l1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("cmnkjl,mnibda->ijklabcd", W_vooooo[..., k0:k1, j0:j1, l0:l1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("dmnlij,mnkabc->ijklabcd", W_vooooo[..., l0:l1, i0:i1, j0:j1],
                        t3[:, :, k0:k1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("dmnlik,mnjacb->ijklabcd", W_vooooo[..., l0:l1, i0:i1, k0:k1],
                        t3[:, :, j0:j1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("dmnljk,mnibca->ijklabcd", W_vooooo[..., l0:l1, j0:j1, k0:k1],
                        t3[:, :, i0:i1], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    einsum("mlcd,abmijk->ijklabcd", t2[:, l0:l1], W_vvoooo[..., i0:i1, j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mkdc,abmijl->ijklabcd", t2[:, k0:k1], W_vvoooo[..., i0:i1, j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mlbd,acmikj->ijklabcd", t2[:, l0:l1], W_vvoooo[..., i0:i1, k0:k1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mkbc,admilj->ijklabcd", t2[:, k0:k1], W_vvoooo[..., i0:i1, l0:l1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mjdb,acmikl->ijklabcd", t2[:, j0:j1], W_vvoooo[..., i0:i1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mjcb,admilk->ijklabcd", t2[:, j0:j1], W_vvoooo[..., i0:i1, l0:l1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mlad,bcmjki->ijklabcd", t2[:, l0:l1], W_vvoooo[..., j0:j1, k0:k1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mkac,bdmjli->ijklabcd", t2[:, k0:k1], W_vvoooo[..., j0:j1, l0:l1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mjab,cdmkli->ijklabcd", t2[:, j0:j1], W_vvoooo[..., k0:k1, l0:l1, i0:i1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mida,bcmjkl->ijklabcd", t2[:, i0:i1], W_vvoooo[..., j0:j1, k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("mica,bdmjlk->ijklabcd", t2[:, i0:i1], W_vvoooo[..., j0:j1, l0:l1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    einsum("miba,cdmklj->ijklabcd", t2[:, i0:i1], W_vvoooo[..., k0:k1, l0:l1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    einsum("iled,abcejk->ijklabcd", t2[i0:i1, l0:l1], W_vvvvoo[..., j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("ikec,abdejl->ijklabcd", t2[i0:i1, k0:k1], W_vvvvoo[..., j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("ijeb,acdekl->ijklabcd", t2[i0:i1, j0:j1], W_vvvvoo[..., k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("jled,baceik->ijklabcd", t2[j0:j1, l0:l1], W_vvvvoo[..., i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("jkec,badeil->ijklabcd", t2[j0:j1, k0:k1], W_vvvvoo[..., i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("kled,cabeij->ijklabcd", t2[k0:k1, l0:l1], W_vvvvoo[..., i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("lkec,dabeij->ijklabcd", t2[l0:l1, k0:k1], W_vvvvoo[..., i0:i1, j0:j1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("kjeb,cadeil->ijklabcd", t2[k0:k1, j0:j1], W_vvvvoo[..., i0:i1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("ljeb,daceik->ijklabcd", t2[l0:l1, j0:j1], W_vvvvoo[..., i0:i1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("jiea,bcdekl->ijklabcd", t2[j0:j1, i0:i1], W_vvvvoo[..., k0:k1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("kiea,cbdejl->ijklabcd", t2[k0:k1, i0:i1], W_vvvvoo[..., j0:j1, l0:l1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    einsum("liea,dbcejk->ijklabcd", t2[l0:l1, i0:i1], W_vvvvoo[..., j0:j1, k0:k1],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    einsum("ae,ijklebcd->ijklabcd", F_vv, t4_tmp[:bi, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, j0, j1, i0, i1, k0, k1, l0, l1)
                    einsum("be,jikleacd->ijklabcd", F_vv, t4_tmp[:bj, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, k0, k1, i0, i1, j0, j1, l0, l1)
                    einsum("ce,kijleabd->ijklabcd", F_vv, t4_tmp[:bk, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, l0, l1, i0, i1, j0, j1, k0, k1)
                    einsum("de,lijkeabc->ijklabcd", F_vv, t4_tmp[:bl, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    einsum("abef,ijklefcd->ijklabcd", W_vvvv, t4_tmp[:bi, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, k0, k1, j0, j1, l0, l1)
                    einsum("acef,ikjlefbd->ijklabcd", W_vvvv, t4_tmp[:bi, :bk, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, i0, i1, l0, l1, j0, j1, k0, k1)
                    einsum("adef,iljkefbc->ijklabcd", W_vvvv, t4_tmp[:bi, :bl, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, j0, j1, k0, k1, i0, i1, l0, l1)
                    einsum("bcef,jkilefad->ijklabcd", W_vvvv, t4_tmp[:bj, :bk, :bi, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, j0, j1, l0, l1, i0, i1, k0, k1)
                    einsum("bdef,jlikefac->ijklabcd", W_vvvv, t4_tmp[:bj, :bl, :bi, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                    _unpack_t4_(mycc, t4, t4_tmp, k0, k1, l0, l1, i0, i1, j0, j1)
                    einsum("cdef,klijefab->ijklabcd", W_vvvv, t4_tmp[:bk, :bl, :bi, :bj],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
        time2 = log.timer_debug1('t4: iter: W_vvoooo * t2, W_vvvvoo * t2,\n'
            '                           W_vvvo * t3, W_vooo * t3, W_ovvvoo * t3, W_ovvovo * t3, W_vooooo * t3,\n'
            '                           F_vv * t4, W_vvvv * t4 [%3d, %3d]:' % (l0, l1), *time2)
    t4_tmp = None
    r4_tmp = None
    c_t3 = None
    F_vv = imds.F_vv = None
    W_vvvo = imds.W_vvvo = None
    W_vooo = imds.W_vooo = None
    W_vvvv = imds.W_vvvv = None
    W_ovvvoo = imds.W_ovvvoo = None
    W_ovvovo = imds.W_ovvovo = None
    W_vooooo = imds.W_vooooo = None
    W_vvoooo = imds.W_vvoooo = None
    W_vvvvoo = imds.W_vvvvoo = None

    time1 = log.timer_debug1('t4: W_vvoooo * t2, W_vvvvoo * t2, W_vvvo * t3, W_vooo * t3, W_ovvvoo * t3,\n'
                        '                     W_ovvovo * t3, W_vooooo * t3, F_vv * t4, W_vvvv * t4', *time1)

    time2 = logger.process_clock(), logger.perf_counter()
    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0

                    r4_tmp[:] = 0.0
                    for m0, m1 in lib.prange(0, nocc, blksize):
                        bm = m1 - m0

                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, k0, k1, l0, l1)
                        einsum("mi,mjklabcd->ijklabcd", F_oo[m0:m1, i0:i1], t4_tmp[:bm, :bj, :bk, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum("maei,mjklebcd->ijklabcd", W_ovvo[m0:m1, :, :, i0:i1],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, k0, k1, l0, l1)
                        einsum("mj,miklbacd->ijklabcd", F_oo[m0:m1, j0:j1], t4_tmp[:bm, :bi, :bk, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum("mbej,mikleacd->ijklabcd", W_ovvo[m0:m1, :, :, j0:j1],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, l0, l1)
                        einsum("mk,mijlcabd->ijklabcd", F_oo[m0:m1, k0:k1], t4_tmp[:bm, :bi, :bj, :bl],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum("mcek,mijleabd->ijklabcd", W_ovvo[m0:m1, :, :, k0:k1],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                        einsum("ml,mijkdabc->ijklabcd", F_oo[m0:m1, l0:l1], t4_tmp[:bm, :bi, :bj, :bk],
                            out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        t4_spin_summation_inplace_(t4_tmp, blksize**4, nvir, "P4_201", 1.0, 0.0)
                        einsum("mdel,mijkeabc->ijklabcd", W_ovvo[m0:m1, :, :, l0:l1],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)

                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, k0, k1, l0, l1)
                        einsum("maie,mjklbecd->ijklabcd", W_ovov[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mbie,mjklaecd->ijklabcd", W_ovov[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, k0, k1, j0, j1, l0, l1)
                        einsum("maie,mkjlcebd->ijklabcd", W_ovov[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bk, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mcie,mkjlaebd->ijklabcd", W_ovov[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bk, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, l0, l1, j0, j1, k0, k1)
                        einsum("maie,mljkdebc->ijklabcd", W_ovov[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bl, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mdie,mljkaebc->ijklabcd", W_ovov[m0:m1, :, i0:i1, :],
                            t4_tmp[:bm, :bl, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, k0, k1, l0, l1)
                        einsum("mbje,miklaecd->ijklabcd", W_ovov[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("maje,miklbecd->ijklabcd", W_ovov[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, l0, l1)
                        einsum("mcke,mijlaebd->ijklabcd", W_ovov[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("make,mijlcebd->ijklabcd", W_ovov[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, i0, i1, j0, j1, k0, k1)
                        einsum("mdle,mijkaebc->ijklabcd", W_ovov[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("male,mijkdebc->ijklabcd", W_ovov[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, k0, k1, i0, i1, l0, l1)
                        einsum("mbje,mkilcead->ijklabcd", W_ovov[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bk, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mcje,mkilbead->ijklabcd", W_ovov[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bk, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, l0, l1, i0, i1, k0, k1)
                        einsum("mbje,mlikdeac->ijklabcd", W_ovov[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bl, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mdje,mlikbeac->ijklabcd", W_ovov[m0:m1, :, j0:j1, :],
                            t4_tmp[:bm, :bl, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, i0, i1, l0, l1)
                        einsum("mcke,mjilbead->ijklabcd", W_ovov[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bj, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mbke,mjilcead->ijklabcd", W_ovov[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bj, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, j0, j1, i0, i1, k0, k1)
                        einsum("mdle,mjikbeac->ijklabcd", W_ovov[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bj, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mble,mjikdeac->ijklabcd", W_ovov[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bj, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, l0, l1, i0, i1, j0, j1)
                        einsum("mcke,mlijdeab->ijklabcd", W_ovov[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bl, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mdke,mlijceab->ijklabcd", W_ovov[m0:m1, :, k0:k1, :],
                            t4_tmp[:bm, :bl, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, m0, m1, k0, k1, i0, i1, j0, j1)
                        einsum("mdle,mkijceab->ijklabcd", W_ovov[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bk, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                        einsum("mcle,mkijdeab->ijklabcd", W_ovov[m0:m1, :, l0:l1, :],
                            t4_tmp[:bm, :bk, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)
        time2 = log.timer_debug1('t4: iter: F_oo * t4, W_ovvo * t4, W_ovov * t4 [%3d, %3d]:'%(l0, l1), *time2)
    t4_tmp = None
    r4_tmp = None
    F_oo = imds.F_oo = None
    W_ovvo = imds.W_ovvo = None
    W_ovov = imds.V_ovov = None
    time1 = log.timer_debug1('t4: F_oo * t4, W_ovvo * t4, W_ovov * t4', *time1)

    time2 = logger.process_clock(), logger.perf_counter()
    t4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0

                    r4_tmp[:] = 0.0
                    for m0, m1 in lib.prange(0, nocc, blksize):
                        bm = m1 - m0
                        for n0, n1 in lib.prange(0, nocc, blksize):
                            bn = n1 - n0

                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, k0, k1, l0, l1)
                            einsum("mnij,mnklabcd->ijklabcd", W_oooo[m0:m1, n0:n1, i0:i1, j0:j1],
                                t4_tmp[:bm, :bn, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, j0, j1, l0, l1)
                            einsum("mnik,mnjlacbd->ijklabcd", W_oooo[m0:m1, n0:n1, i0:i1, k0:k1],
                                t4_tmp[:bm, :bn, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, j0, j1, k0, k1)
                            einsum("mnil,mnjkadbc->ijklabcd", W_oooo[m0:m1, n0:n1, i0:i1, l0:l1],
                                t4_tmp[:bm, :bn, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, l0, l1)
                            einsum("mnjk,mnilbcad->ijklabcd", W_oooo[m0:m1, n0:n1, j0:j1, k0:k1],
                                t4_tmp[:bm, :bn, :bi, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, k0, k1)
                            einsum("mnjl,mnikbdac->ijklabcd", W_oooo[m0:m1, n0:n1, j0:j1, l0:l1],
                                t4_tmp[:bm, :bn, :bi, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                            _unpack_t4_(mycc, t4, t4_tmp, m0, m1, n0, n1, i0, i1, j0, j1)
                            einsum("mnkl,mnijcdab->ijklabcd", W_oooo[m0:m1, n0:n1, k0:k1, l0:l1],
                                t4_tmp[:bm, :bn, :bi, :bj], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)
        time2 = log.timer_debug1('t4: iter: W_oooo * t4 [%3d, %3d]:'%(l0, l1), *time2)
    t4_tmp = None
    r4_tmp = None
    W_oooo = imds.W_oooo = None
    time1 = log.timer_debug1('t4: W_oooo * t4', *time1)
    return r4

def r4_tri_divide_e_(mycc, r4, mo_energy):
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    eia = mo_energy[: nocc, None] - mo_energy[None, nocc :] - mycc.level_shift
    eijklabcd_blk = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=r4.dtype)
    r4_tmp = np.empty((blksize,) * 4 + (nvir,) * 4, dtype=r4.dtype)
    for l0, l1 in lib.prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in lib.prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in lib.prange(0, j1, blksize):
                    bi = i1 - i0
                    eijklabcd_blk = (eia[i0:i1, None, None, None, :, None, None, None]
                                + eia[None, j0:j1, None, None, None, :, None, None]
                                + eia[None, None, k0:k1, None, None, None, :, None]
                                + eia[None, None, None, l0:l1, None, None, None, :])
                    _unpack_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
                    r4_tmp[:bi, :bj, :bk, :bl] /= eijklabcd_blk
                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1)
    eijklabcd_blk = None
    r4_tmp = None
    return r4

def update_amps_rccsdtq_tri_(mycc, tamps, eris):
    '''Update RCCSDTQ amplitudes in place, with T4 amplitudes stored in triangular form.'''
    assert (isinstance(eris, _PhysicistsERIs))
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1, t2, t3, t4 = tamps
    nocc4 = t4.shape[0]
    mo_energy = eris.mo_energy

    imds = _IMDS()

    # t1, t2
    update_t1_fock_eris(mycc, imds, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2(mycc, imds, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, imds, t2)
    r1r2_add_t3_(mycc, imds, r1, r2, t3)
    r2_add_t4_tri_(mycc, imds, r2, t4)
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
    r3_add_t4_tri_(mycc, imds, r3, t4)
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
    intermediates_t4_tri(mycc, imds, t2, t3, t4)
    imds.t1_fock, imds.t1_eris = None, None
    time1 = log.timer_debug1('t4: update intermediates', *time0)
    r4 = compute_r4_tri(mycc, imds, t2, t3, t4)
    imds = None
    time1 = log.timer_debug1('t4: compute r4', *time1)
    # symmetrization
    symmetrize_tamps_tri_(r4, nocc)
    t4_spin_summation_inplace_(r4, nocc4, nvir, "P4_full", -1.0 / 24.0, 1.0)
    purify_tamps_tri_(r4, nocc)
    time1 = log.timer_debug1('t4: symmetrize r4', *time1)
    # divide by eijkabc
    r4_tri_divide_e_(mycc, r4, mo_energy)
    time1 = log.timer_debug1('t4: divide r4 by eijklabcd', *time1)

    res_norm.append(np.linalg.norm(r4))

    # t4 += r4
    t4_add_(t4, r4, nocc4, nvir)
    r4 = None
    time1 = log.timer_debug1('t4: update t4', *time1)
    time0 = log.timer_debug1('t4 total', *time0)
    return res_norm

def memory_estimate_log_rccsdtq(mycc):
    '''Estimate the memory cost.'''
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    log.info('Approximate memory usage estimate')
    if mycc.do_tri_max_t:
        nocc4 = nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24
        t4_memory = nocc4 * nvir**4 * 8
    else:
        t4_memory = nocc**4 * nvir**4 * 8
    log.info('    T4 memory               %8s', format_size(t4_memory))
    log.info('    R4 memory               %8s', format_size(t4_memory))

    if not mycc.do_tri_max_t:
        symm_t4_memory = t4_memory
        log.info('    Symmetrized T4 memory   %8s', format_size(symm_t4_memory))
    if mycc.do_tri_max_t:
        if nocc * (nocc + 1) * (nocc + 2) // 6 >= 100:
            factor = 4
        else:
            factor = 1
        symm_t4_memory = nocc * (nocc + 1) * (nocc + 2) // 6 * nvir**4 * 8 * 2 / factor
        log.info('    Symmetrized T4 memory   %8s', format_size(symm_t4_memory))

    eris_memory = nmo**4 * 8
    log.info('    ERIs memory             %8s', format_size(eris_memory))
    log.info('    T1-ERIs memory          %8s', format_size(eris_memory))
    intermediates_memory = nocc**2 * nvir**4 * 8 * 2
    log.info('    Intermediates memory    %8s', format_size(intermediates_memory))

    if mycc.do_tri_max_t:
        blk_memory = mycc.blksize**4 * nvir**4 * 8 * 2
        log.info("    Block workspace         %8s", format_size(blk_memory))

    if mycc.einsum_backend in ['numpy', 'pyscf']:
        if mycc.do_tri_max_t:
            einsum_memory = blk_memory
            log.info("    T4 einsum buffer        %8s", format_size(einsum_memory))
        else:
            einsum_memory = t4_memory
            log.info("    T4 einsum buffer        %8s", format_size(einsum_memory))

    if mycc.incore_complete:
        if mycc.do_diis_max_t:
            diis_memory = nocc * (nocc + 1) * (nocc + 2) * (nocc + 3) // 24 * nvir**4 * 8  * mycc.diis_space * 2
        else:
            diis_memory = nocc * (nocc + 1) * (nocc + 2) // 6 * nvir**3 * 8 * mycc.diis_space * 2
        log.info('    DIIS memory             %8s', format_size(diis_memory))
    else:
        diis_memory = 0.0

    total_memory = 2 * t4_memory + symm_t4_memory + 3 * eris_memory + diis_memory
    if mycc.do_tri_max_t:
        total_memory += blk_memory
    if mycc.einsum_backend in ['numpy', 'pyscf']:
        total_memory += einsum_memory

    log.info('Total estimated memory      %8s', format_size(total_memory))
    max_memory = mycc.max_memory - lib.current_memory()[0]
    if (total_memory / 1024**2) > max_memory:
        logger.warn(mycc, 'Estimated memory usage exceeds the allowed limit for %s', mycc.__class__.__name__)
        logger.warn(mycc, 'The calculation may run out of memory')
        if mycc.incore_complete:
            if mycc.do_diis_max_t:
                logger.warn(mycc, 'Consider setting `do_diis_max_t = False` to reduce memory usage')
            else:
                logger.warn(mycc, 'Consider setting `incore_complete = False` to reduce memory usage')
        if not mycc.do_tri_max_t:
            logger.warn(mycc, 'Consider using %s in `pyscf.cc.rccsdtq` which stores the triangular T amplitudes',
                        mycc.__class__.__name__)
        else:
            logger.warn(mycc, 'Consider reducing `blksize` to reduce memory usage')
    return mycc

def dump_chk(mycc, tamps=None, frozen=None, mo_coeff=None, mo_occ=None):
    if not mycc.chkfile:
        return mycc
    if tamps is None: tamps = mycc.tamps
    if frozen is None: frozen = mycc.frozen
    # "None" cannot be serialized by the chkfile module
    if frozen is None:
        frozen = 0
    cc_chk = {'e_corr': mycc.e_corr, 'tamps': tamps, 'frozen': frozen}
    if mo_coeff is not None: cc_chk['mo_coeff'] = mo_coeff
    if mo_occ is not None: cc_chk['mo_occ'] = mo_occ
    if mycc._nmo is not None: cc_chk['_nmo'] = mycc._nmo
    if mycc._nocc is not None: cc_chk['_nocc'] = mycc._nocc
    if mycc.do_tri_max_t:
        lib.chkfile.save(mycc.chkfile, 'rccsdtq', cc_chk)
    else:
        lib.chkfile.save(mycc.chkfile, 'rccsdtq_highm', cc_chk)


class RCCSDTQ(rccsdt.RCCSDT):
    __doc__ = f'''{__doc__}
Attributes such as conv_tol, max_cycle, diis_space, diis_start_cycle,
iterative_damping, incore_complete, level_shift, and frozen can be configured in
the same way as in CCSD. Additional attributes are:

    do_diis_max_t : bool
        Whether to use DIIS to accelerate convergence. Note that enabling DIIS
        will increase memory consumption.
    blksize, blksize_oovv, blksize_oooo :
        Batch sizes used to reduce the memory footprint during tensor contractions.
    einsum_backend : string
        Selects a more efficient einsum backend, such as pytblis or PySCF
        built-in einsum. By default, numpy.einsum is used.

Saved results:

    converged : bool
        Whether the CCSDT iteration converged
    e_corr : float
        CCSDT correlation correction
    e_tot : float
        Total CCSDT energy (HF + correlation)
    cycles : int
        Number of iteration cycles performed.
    t1, t2, t3 :
        T amplitudes t1[i,a], t2[i,j,a,b], t3[i,j,k,a,b,c]
    t4 :
        An array of shape (compressed_occ, nvir, nvir, nvir, nvir) for T4 amplitudes.
        The occupied-oribtal dimension is stored in a compressed form for the
        i <= j <= k <= l index combinations. The compressed tensor can be expanded to
        the full tensor by self.tamps_tri2full(t4)
    tamps :
        A tuple (t1, t2, t3, t4) containing the RCCSDTQ cluster amplitudes.
'''

    conv_tol = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_conv_tol_normt', 1e-6)
    cc_order = 4
    do_diis_max_t = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_do_diis_max_t', True)
    blksize = getattr(__config__, 'cc_rccsdtq_RCCSDTQ_blksize', 4)

    @property
    def t4(self):
        return self.tamps[3]

    @t4.setter
    def t4(self, val):
        self.tamps[3] = val

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        rccsdt.RCCSDT.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.tamps = [None, None, None, None]

    do_tri_max_t = property(lambda self: True)

    memory_estimate_log = memory_estimate_log_rccsdtq
    update_amps_ = update_amps_rccsdtq_tri_
    dump_chk = dump_chk

    def kernel(self, tamps=None, eris=None):
        return self.ccsdtq(tamps, eris)

    def ccsdtq(self, tamps=None, eris=None):
        log = logger.Logger(self.stdout, self.verbose)

        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        assert self.mo_coeff.dtype == np.float64, "`mo_coeff` must be float64"

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        nocc = eris.nocc

        if self.do_tri_max_t:
            self.tri2block_map, self.tri2block_mask, self.tri2block_tp = self.setup_tri2block()

            self.blksize = min(self.blksize, nocc)
            log.info('blksize %2d' % (self.blksize))
            if self.blksize > (nocc + 1) // 2:
                logger.warn(self, 'A large `blksize` is being used, which may cause large memory consumption\n'
                            '      for storing contraction intermediates. If memory is sufficient, consider using\n'
                            '      `pyscf.cc.rccsdtq_highm.RCCSDTQ` instead.')

        self.memory_estimate_log()
        self.unique_tamps_map = self.build_unique_tamps_map()

        self.converged, self.e_corr, self.tamps = kernel(self, eris, tamps, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt, verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.tamps

class _IMDS:

    def __init__(self):
        self.t1_fock = None
        self.t1_eris = None
        self.F_oo = None
        self.F_vv = None
        self.W_oooo = None
        self.W_ovvo = None
        self.W_ovov = None
        self.W_vooo = None
        self.W_vvvo = None
        self.W_vvvv = None
        self.W_ovvvoo = None
        self.W_ovvovo = None
        self.W_vooooo = None
        self.W_vvoooo = None
        self.W_vvvvoo = None


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
    print('\n' * 5)

    # comparison with the high-memory version
    from pyscf.cc.rccsdtq_highm import RCCSDTQ as RCCSDTQhm
    mycc2 = RCCSDTQhm(mf, frozen=frozen)
    mycc2.set_einsum_backend('numpy')
    mycc2.conv_tol = 1e-12
    mycc2.conv_tol_normt = 1e-10
    mycc2.max_cycle = 100
    mycc2.verbose = 5
    mycc2.do_diis_max_t = True
    mycc2.incore_complete = True
    mycc2.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc2.e_corr, ref_ecorr, mycc2.e_corr - ref_ecorr))
    print()

    t4_tri = mycc.t4
    t4_full = mycc2.t4
    t4_tri_from_t4_full = mycc2.tamps_full2tri(t4_full)
    t4_full_from_t4_tri = mycc.tamps_tri2full(t4_tri)

    print('energy difference                          % .10e' % (mycc.e_tot - mycc2.e_tot))
    print('max(abs(t1 difference))                    % .10e' % np.max(np.abs(mycc.t1 - mycc2.t1)))
    print('max(abs(t2 difference))                    % .10e' % np.max(np.abs(mycc.t2 - mycc2.t2)))
    print('max(abs(t3 difference))                    % .10e' % np.max(np.abs(mycc.t3 - mycc2.t3)))
    print('max(abs(t4_tri - t4_tri_from_t4_full))     % .10e' % np.max(np.abs(t4_tri - t4_tri_from_t4_full)))
    print('max(abs(t4_full - t4_full_from_t4_tri))    % .10e' % np.max(np.abs(t4_full - t4_full_from_t4_tri)))
