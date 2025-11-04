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
UHF-CCSDT with symmetry-unique T3 storage:
    - t3_aaa, t3_bbb:  i < j < k, a < b < c
    - t3_aab, t3_bba:  i < j, a < b

T2 amplitudes are stored as t2aa, t2ab, and t2bb, where t2ab has the shape (nocca, nvira, noccb, nvirb).
This differs from the convention in `pyscf.cc.uccsd`, where t2ab is stored as (nocca, noccb, nvira, nvirb).

Equations derived from the GCCSDT equations in Shavitt and Bartlett, Many-Body Methods in Chemistry and Physics:
MBPT and Coupled-Cluster Theory, Cambridge University Press (2009). DOI: 10.1017/CBO9780511596834.
'''

import numpy as np
import numpy
from functools import reduce
import functools
import ctypes
from pyscf import ao2mo, lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_e_hf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.cc import ccsd, _ccsd
from pyscf.cc.rccsdt import _einsum, run_diis, _finalize, dump_flags, kernel, format_size
from pyscf import __config__


_libccsdt = lib.load_library('libccsdt')

def unpack_t3_aaa_tri2block_(t3, t3_blk, map_o, mask_o, map_v, mask_v, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                                nocc, nvir, blk_i, blk_j, blk_k, blk_a, blk_b, blk_c):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and mask_o.dtype == np.bool_
    assert map_v.dtype == np.int64 and mask_v.dtype == np.bool_
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_o = np.ascontiguousarray(map_o)
    mask_o = np.ascontiguousarray(mask_o)
    map_v = np.ascontiguousarray(map_v)
    mask_v = np.ascontiguousarray(mask_v)
    _libccsdt.unpack_t3_aaa_tri2block_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_o.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        map_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_v.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b), ctypes.c_int64(blk_c)
    )
    return t3_blk

def accumulate_t3_aaa_block2tri_(t3, t3_blk, map_o, map_v, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                                    nocc, nvir, blk_i, blk_j, blk_k, blk_a, blk_b, blk_c, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and map_v.dtype == np.int64
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_o = np.ascontiguousarray(map_o)
    map_v = np.ascontiguousarray(map_v)
    _libccsdt.accumulate_t3_aaa_block2tri_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        map_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b), ctypes.c_int64(blk_c),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def unpack_t3_aab_tri2block_(t3, t3_blk, map_o, mask_o, map_v, mask_v, i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                nocc, nvir, dim4, dim5, blk_i, blk_j, blk_a, blk_b):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and mask_o.dtype == np.bool_
    assert map_v.dtype == np.int64 and mask_v.dtype == np.bool_
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_o = np.ascontiguousarray(map_o)
    mask_o = np.ascontiguousarray(mask_o)
    map_v = np.ascontiguousarray(map_v)
    mask_v = np.ascontiguousarray(mask_v)
    _libccsdt.unpack_t3_aab_tri2block_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_o.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        map_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask_v.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(dim4), ctypes.c_int64(dim5),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b),
    )
    return t3_blk

def accumulate_t3_aab_block2tri_(t3, t3_blk, map_o, map_v, i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                    nocc, nvir, dim4, dim5, blk_i, blk_j, blk_a, blk_b, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_o.dtype == np.int64 and map_v.dtype == np.int64
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_o = np.ascontiguousarray(map_o)
    map_v = np.ascontiguousarray(map_v)
    _libccsdt.accumulate_t3_aab_block2tri_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_o.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        map_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(a0), ctypes.c_int64(a1),
        ctypes.c_int64(b0), ctypes.c_int64(b1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(c0), ctypes.c_int64(c1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(dim4), ctypes.c_int64(dim5),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j),
        ctypes.c_int64(blk_a), ctypes.c_int64(blk_b),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def _tri2block_6fold(no):
    no3 = no * (no - 1) * (no - 2) // 6
    tri2block_map = np.zeros((6, no, no, no), dtype=np.int64)
    tri2block_mask = np.zeros((6, no, no, no), dtype=np.bool_)
    i, j, k = np.meshgrid(np.arange(no), np.arange(no), np.arange(no), indexing='ij')
    t3_map = np.where((i < j) & (j < k))
    tri2block_map[0, t3_map[0], t3_map[1], t3_map[2]] = np.arange(no3)
    tri2block_map[1, t3_map[0], t3_map[2], t3_map[1]] = np.arange(no3)
    tri2block_map[2, t3_map[1], t3_map[0], t3_map[2]] = np.arange(no3)
    tri2block_map[3, t3_map[1], t3_map[2], t3_map[0]] = np.arange(no3)
    tri2block_map[4, t3_map[2], t3_map[0], t3_map[1]] = np.arange(no3)
    tri2block_map[5, t3_map[2], t3_map[1], t3_map[0]] = np.arange(no3)
    tri2block_mask[0] = (i < j) & (j < k)
    tri2block_mask[1] = (i < k) & (k < j)
    tri2block_mask[2] = (j < i) & (i < k)
    tri2block_mask[3] = (k < i) & (i < j)
    tri2block_mask[4] = (j < k) & (k < i)
    tri2block_mask[5] = (k < j) & (j < i)
    return tri2block_map, tri2block_mask

def _tri2block_2fold(no):
    no2 = no * (no - 1) // 2
    tri2block_map = np.zeros((2, no, no), dtype=np.int64)
    tri2block_mask = np.zeros((2, no, no), dtype=np.bool_)
    i, j = np.meshgrid(np.arange(no), np.arange(no), indexing='ij')
    t3_map = np.where(i < j)
    tri2block_map[0, t3_map[0], t3_map[1]] = np.arange(no2)
    tri2block_map[1, t3_map[1], t3_map[0]] = np.arange(no2)
    tri2block_mask[0] = i < j
    tri2block_mask[1] = i > j
    return tri2block_map, tri2block_mask

def setup_tri2block_t3_uhf(mycc):
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    t2c_map_6f_oa, t2c_mask_6f_oa = _tri2block_6fold(nocca)
    t2c_map_2f_oa, t2c_mask_2f_oa = _tri2block_2fold(nocca)
    t2c_map_6f_va, t2c_mask_6f_va = _tri2block_6fold(nvira)
    t2c_map_2f_va, t2c_mask_2f_va = _tri2block_2fold(nvira)
    t2c_map_6f_ob, t2c_mask_6f_ob = _tri2block_6fold(noccb)
    t2c_map_2f_ob, t2c_mask_2f_ob = _tri2block_2fold(noccb)
    t2c_map_6f_vb, t2c_mask_6f_vb = _tri2block_6fold(nvirb)
    t2c_map_2f_vb, t2c_mask_2f_vb = _tri2block_2fold(nvirb)
    return (t2c_map_6f_oa, t2c_mask_6f_oa, t2c_map_2f_oa, t2c_mask_2f_oa, t2c_map_6f_va, t2c_mask_6f_va,
            t2c_map_2f_va, t2c_mask_2f_va, t2c_map_6f_ob, t2c_mask_6f_ob, t2c_map_2f_ob, t2c_mask_2f_ob,
            t2c_map_6f_vb, t2c_mask_6f_vb, t2c_map_2f_vb, t2c_mask_2f_vb)

def _unp_aaa_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
            blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None):
    nocca, nmoa = mycc.nocc[0], mycc.nmo[0]
    nvira = nmoa - nocca
    if not blk_i: blk_i = mycc.blksize_o_aaa
    if not blk_j: blk_j = mycc.blksize_o_aaa
    if not blk_k: blk_k = mycc.blksize_o_aaa
    if not blk_a: blk_a = mycc.blksize_v_aaa
    if not blk_b: blk_b = mycc.blksize_v_aaa
    if not blk_c: blk_c = mycc.blksize_v_aaa
    unpack_t3_aaa_tri2block_(t3, t3_blk, mycc.t2c_map_6f_oa, mycc.t2c_mask_6f_oa,
                            mycc.t2c_map_6f_va, mycc.t2c_mask_6f_va, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                            nocca, nvira, blk_i, blk_j, blk_k, blk_a, blk_b, blk_c)
    return t3_blk

def _unp_bbb_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
            blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None):
    noccb, nmob = mycc.nocc[1], mycc.nmo[1]
    nvirb = nmob - noccb
    if not blk_i: blk_i = mycc.blksize_o_aaa
    if not blk_j: blk_j = mycc.blksize_o_aaa
    if not blk_k: blk_k = mycc.blksize_o_aaa
    if not blk_a: blk_a = mycc.blksize_v_aaa
    if not blk_b: blk_b = mycc.blksize_v_aaa
    if not blk_c: blk_c = mycc.blksize_v_aaa
    unpack_t3_aaa_tri2block_(t3, t3_blk, mycc.t2c_map_6f_ob, mycc.t2c_mask_6f_ob,
                            mycc.t2c_map_6f_vb, mycc.t2c_mask_6f_vb, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                            noccb, nvirb, blk_i, blk_j, blk_k, blk_a, blk_b, blk_c)
    return t3_blk

def _update_packed_aaa_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                        blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None, alpha=1.0, beta=0.0):
    nocca, nmoa = mycc.nocc[0], mycc.nmo[0]
    nvira = nmoa - nocca
    if not blk_i: blk_i = mycc.blksize_o_aaa
    if not blk_j: blk_j = mycc.blksize_o_aaa
    if not blk_k: blk_k = mycc.blksize_o_aaa
    if not blk_a: blk_a = mycc.blksize_v_aaa
    if not blk_b: blk_b = mycc.blksize_v_aaa
    if not blk_c: blk_c = mycc.blksize_v_aaa
    accumulate_t3_aaa_block2tri_(t3, t3_blk, mycc.t2c_map_6f_oa, mycc.t2c_map_6f_va,
        i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1, nocca, nvira,
        blk_i, blk_j, blk_k, blk_a, blk_b, blk_c, alpha=alpha, beta=beta)
    return t3

def _update_packed_bbb_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1,
                        blk_i=None, blk_j=None, blk_k=None, blk_a=None, blk_b=None, blk_c=None, alpha=1.0, beta=0.0):
    noccb, nmob = mycc.nocc[1], mycc.nmo[1]
    nvirb = nmob - noccb
    if not blk_i: blk_i = mycc.blksize_o_aaa
    if not blk_j: blk_j = mycc.blksize_o_aaa
    if not blk_k: blk_k = mycc.blksize_o_aaa
    if not blk_a: blk_a = mycc.blksize_v_aaa
    if not blk_b: blk_b = mycc.blksize_v_aaa
    if not blk_c: blk_c = mycc.blksize_v_aaa
    accumulate_t3_aaa_block2tri_(t3, t3_blk, mycc.t2c_map_6f_ob, mycc.t2c_map_6f_vb,
        i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1, noccb, nvirb,
        blk_i, blk_j, blk_k, blk_a, blk_b, blk_c, alpha=alpha, beta=beta)
    return t3

def _unp_aab_(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1, k0=None, k1=None, c0=None, c1=None,
                blk_i=None, blk_j=None, blk_a=None, blk_b=None, dim4=None, dim5=None):
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    if not k0: k0 = 0
    if not k1: k1 = noccb
    if not c0: c0 = 0
    if not c1: c1 = nvirb
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = noccb
    if not dim5: dim5 = nvirb
    unpack_t3_aab_tri2block_(t3, t3_blk, mycc.t2c_map_2f_oa, mycc.t2c_mask_2f_oa,
                                mycc.t2c_map_2f_va, mycc.t2c_mask_2f_va,
                                i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                nocca, nvira, dim4, dim5, blk_i, blk_j, blk_a, blk_b)
    return t3_blk

def _unp_bba_(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1, k0=None, k1=None, c0=None, c1=None,
                blk_i=None, blk_j=None, blk_a=None, blk_b=None, dim4=None, dim5=None):
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    if not k0: k0 = 0
    if not k1: k1 = nocca
    if not c0: c0 = 0
    if not c1: c1 = nvira
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = nocca
    if not dim5: dim5 = nvira
    unpack_t3_aab_tri2block_(t3, t3_blk, mycc.t2c_map_2f_ob, mycc.t2c_mask_2f_ob,
                                mycc.t2c_map_2f_vb, mycc.t2c_mask_2f_vb,
                                i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1,
                                noccb, nvirb, dim4, dim5, blk_i, blk_j, blk_a, blk_b)
    return t3_blk

def _update_packed_aab_(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1, k0=None, k1=None, c0=None, c1=None,
                        blk_i=None, blk_j=None, blk_a=None, blk_b=None, dim4=None, dim5=None, alpha=1.0, beta=0.0):
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    if not k0: k0 = 0
    if not k1: k1 = noccb
    if not c0: c0 = 0
    if not c1: c1 = nvirb
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = noccb
    if not dim5: dim5 = nvirb
    accumulate_t3_aab_block2tri_(t3, t3_blk, mycc.t2c_map_2f_oa, mycc.t2c_map_2f_va,
                                    i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1, nocca, nvira, dim4, dim5,
                                    blk_i, blk_j, blk_a, blk_b, alpha=alpha, beta=beta)
    return t3

def _update_packed_bba_(mycc, t3, t3_blk, i0, i1, j0, j1, a0, a1, b0, b1, k0=None, k1=None, c0=None, c1=None,
                        blk_i=None, blk_j=None, blk_a=None, blk_b=None, dim4=None, dim5=None, alpha=1.0, beta=0.0):
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    if not k0: k0 = 0
    if not k1: k1 = nocca
    if not c0: c0 = 0
    if not c1: c1 = nvira
    if not blk_i: blk_i = mycc.blksize_o_aab
    if not blk_j: blk_j = mycc.blksize_o_aab
    if not blk_a: blk_a = mycc.blksize_v_aab
    if not blk_b: blk_b = mycc.blksize_v_aab
    if not dim4: dim4 = nocca
    if not dim5: dim5 = nvira
    accumulate_t3_aab_block2tri_(t3, t3_blk, mycc.t2c_map_2f_ob, mycc.t2c_map_2f_vb,
                                    i0, i1, j0, j1, a0, a1, b0, b1, k0, k1, c0, c1, noccb, nvirb, dim4, dim5,
                                    blk_i, blk_j, blk_a, blk_b, alpha=alpha, beta=beta)
    return t3

def init_amps_uhf(mycc, eris=None):
    '''Initialize CC T-amplitudes for a UHF reference.'''
    time0 = logger.process_clock(), logger.perf_counter()
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    mo_energy = eris.mo_energy[:]
    focka, fockb = eris.fock[0], eris.fock[1]
    nocca, noccb = eris.nocc
    nvira, nvirb = focka.shape[0] - nocca, fockb.shape[0] - noccb

    fova = focka[:nocca, nocca:]
    fovb = fockb[:noccb, noccb:]
    mo_ea_o = mo_energy[0][:nocca]
    mo_ea_v = mo_energy[0][nocca:]
    mo_eb_o = mo_energy[1][:noccb]
    mo_eb_v = mo_energy[1][noccb:]
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    t1a = fova.conj() / eia_a
    t1b = fovb.conj() / eia_b

    t2aa = eris.pppp[:nocca, :nocca, nocca:, nocca:].conj() / lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    t2ab = eris.pPpP[:nocca, :noccb, nocca:, noccb:].conj() / lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    t2bb = eris.PPPP[:noccb, :noccb, noccb:, noccb:].conj() / lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
    # NOTE: The definition of t2ab here differs from the one used in uccsd.py
    t2ab = t2ab.transpose(0, 2, 1, 3)
    t2aa = t2aa - t2aa.transpose(0, 1, 3, 2)
    t2bb = t2bb - t2bb.transpose(0, 1, 3, 2)
    e  =        np.einsum('iaJB,iJaB', t2ab, eris.pPpP[:nocca, :noccb, nocca:, noccb:])
    e += 0.25 * np.einsum('ijab,ijab', t2aa, eris.pppp[:nocca, :nocca, nocca:, nocca:])
    e -= 0.25 * np.einsum('ijab,ijba', t2aa, eris.pppp[:nocca, :nocca, nocca:, nocca:])
    e += 0.25 * np.einsum('ijab,ijab', t2bb, eris.PPPP[:noccb, :noccb, noccb:, noccb:])
    e -= 0.25 * np.einsum('ijab,ijba', t2bb, eris.PPPP[:noccb, :noccb, noccb:, noccb:])
    mycc.emp2 = e.real
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2)

    tamps = [(t1a, t1b), (t2aa, t2ab, t2bb)]
    from math import prod, factorial
    nx = lambda n, order: prod(n - i for i in range(order)) // factorial(order)
    cc_order = mycc.cc_order
    for order in range(2, cc_order - 1):
        t = []
        for na in range(order + 1, -1, -1):
            nb = order + 1 - na
            if na >= nb:
                n1, nocc1, nvir1, n2, nocc2, nvir2 = na, nocca, nvira, nb, noccb, nvirb
            else:
                n1, nocc1, nvir1, n2, nocc2, nvir2 = nb, noccb, nvirb, na, nocca, nvira
            t_ = np.zeros((nocc1,) * (n1) + (nvir1,) * (n1) + (nocc2,) * (n2) + (nvir2,) * (n2), dtype=t1a.dtype)
            t.append(t_)
        tamps.append(t)
    t = []
    for na in range(cc_order, -1, -1):
        nb = cc_order - na
        if na >= nb:
            n1, nocc1, nvir1, n2, nocc2, nvir2 = na, nocca, nvira, nb, noccb, nvirb
        else:
            n1, nocc1, nvir1, n2, nocc2, nvir2 = nb, noccb, nvirb, na, nocca, nvira
        if mycc.do_tri_max_t:
            if n2 == 0:
                shape = (nx(nocc1, n1),) + (nx(nvir1, n1),)
            else:
                shape = (nx(nocc1, n1),) + (nx(nvir1, n1),) + (nx(nocc2, n2),) + (nx(nvir2, n2),)
        else:
            shape = (nocc1,) * (n1) + (nvir1,) * (n1) + (nocc2,) * (n2) + (nvir2,) * (n2)
        t_ = np.zeros(shape, dtype=t1a.dtype)
        t.append(t_)
    tamps.append(t)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, tamps

def energy_uhf(mycc, tamps, eris=None):
    '''CC correlation energy for an RHF reference.'''
    if tamps is None:
        t1, t2 = mycc.tamps[:2]
    else:
        t1, t2 = tamps[:2]
    if eris is None: eris = mycc.ao2mo()

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira, noccb, nvirb = t2ab.shape
    focka, fockb = eris.fock[0], eris.fock[1]
    fova = focka[:nocca, nocca:]
    fovb = fockb[:noccb, noccb:]
    ess  = np.einsum('ia,ia', fova, t1a)
    ess += np.einsum('ia,ia', fovb, t1b)
    ess += 0.25 * np.einsum('ijab,ijab', t2aa, eris.pppp[:nocca, :nocca, nocca:, nocca:])
    ess -= 0.25 * np.einsum('ijab,ijba', t2aa, eris.pppp[:nocca, :nocca, nocca:, nocca:])
    ess += 0.25 * np.einsum('ijab,ijab', t2bb, eris.PPPP[:noccb, :noccb, noccb:, noccb:])
    ess -= 0.25 * np.einsum('ijab,ijba', t2bb, eris.PPPP[:noccb, :noccb, noccb:, noccb:])
    eos  =        np.einsum('iaJB,iJaB', t2ab, eris.pPpP[:nocca, :noccb, nocca:, noccb:])
    ess += 0.5 * lib.einsum('ia,jb,ijab', t1a, t1a, eris.pppp[:nocca, :nocca, nocca:, nocca:])
    ess -= 0.5 * lib.einsum('ia,jb,ijba', t1a, t1a, eris.pppp[:nocca, :nocca, nocca:, nocca:])
    ess += 0.5 * lib.einsum('ia,jb,ijab', t1b, t1b, eris.PPPP[:noccb, :noccb, noccb:, noccb:])
    ess -= 0.5 * lib.einsum('ia,jb,ijba', t1b, t1b, eris.PPPP[:noccb, :noccb, noccb:, noccb:])
    eos +=       lib.einsum('ia,JB,iJaB', t1a, t1b, eris.pPpP[:nocca, :noccb, nocca:, noccb:])

    if abs((ess + eos).imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in %s energy %s', mycc.__class__.name, ess + eos)

    mycc.e_corr = lib.tag_array((ess + eos).real, e_corr_ss=ess.real, e_corr_os=eos.real)
    return mycc.e_corr.real

def update_xy_uhf(mycc, t1):
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    t1a, t1b = t1
    xa = np.eye(nmoa, dtype=t1a.dtype)
    xb = np.eye(nmob, dtype=t1b.dtype)
    xa[nocca:, :nocca] -= t1a.T
    xb[noccb:, :noccb] -= t1b.T
    ya = np.eye(nmoa, dtype=t1a.dtype)
    yb = np.eye(nmob, dtype=t1b.dtype)
    ya[:nocca, nocca:] += t1a
    yb[:noccb, noccb:] += t1b
    return xa, xb, ya, yb

def update_fock_uhf(mycc, xa, xb, ya, yb, t1, eris):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    t1a, t1b = t1
    focka, fockb = eris.focka, eris.fockb
    erisaa, erisab, erisbb = eris.pppp, eris.pPpP, eris.PPPP
    t1_focka = focka + einsum('risa,ia->rs', erisaa[:, :nocca, :, nocca:], t1a)
    t1_focka += einsum('risa,ia->rs', erisab[:, :noccb, :, noccb:], t1b)
    t1_focka -= einsum('rias,ia->rs', erisaa[:, :nocca, nocca:, :], t1a)
    t1_focka = xa @ t1_focka @ ya.T
    t1_fockb = fockb + einsum('risa,ia->rs', erisbb[:, :noccb, :, noccb:], t1b)
    t1_fockb += einsum('iras,ia->rs', erisab[:nocca, :, nocca:, :], t1a)
    t1_fockb -= einsum('rias,ia->rs', erisbb[:, :noccb, noccb:, :], t1b)
    t1_fockb = xb @ t1_fockb @ yb.T
    return t1_focka, t1_fockb

def update_eris_uhf(mycc, xa, xb, ya, yb, eris):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    erisaa, erisab, erisbb = eris.pppp, eris.pPpP, eris.PPPP
    t1_erisaa = einsum('tvuw,pt->pvuw', erisaa, xa)
    t1_erisaa = einsum('pvuw,rv->pruw', t1_erisaa, xa)
    t1_erisaa = t1_erisaa.transpose(2, 3, 0, 1)
    if not t1_erisaa.flags['C_CONTIGUOUS']:
        t1_erisaa = np.ascontiguousarray(t1_erisaa)
    t1_erisaa = einsum('uwpr,qu->qwpr', t1_erisaa, ya)
    t1_erisaa = einsum('qwpr,sw->qspr', t1_erisaa, ya)
    t1_erisaa = t1_erisaa.transpose(2, 3, 0, 1)
    # anti-symmetrization
    t1_erisaa -= t1_erisaa.transpose(0, 1, 3, 2)

    t1_erisbb = einsum('tvuw,pt->pvuw', erisbb, xb)
    t1_erisbb = einsum('pvuw,rv->pruw', t1_erisbb, xb)
    t1_erisbb = t1_erisbb.transpose(2, 3, 0, 1)
    if not t1_erisbb.flags['C_CONTIGUOUS']:
        t1_erisbb = np.ascontiguousarray(t1_erisbb)
    t1_erisbb = einsum('uwpr,qu->qwpr', t1_erisbb, yb)
    t1_erisbb = einsum('qwpr,sw->qspr', t1_erisbb, yb)
    t1_erisbb = t1_erisbb.transpose(2, 3, 0, 1)
    # anti-symmetrization
    t1_erisbb -= t1_erisbb.transpose(0, 1, 3, 2)

    t1_erisab = einsum('tvuw,pt->pvuw', erisab, xa)
    t1_erisab = einsum('pvuw,rv->pruw', t1_erisab, xb)
    t1_erisab = t1_erisab.transpose(2, 3, 0, 1)
    if not t1_erisab.flags['C_CONTIGUOUS']:
        t1_erisab = np.ascontiguousarray(t1_erisab)
    t1_erisab = einsum('uwpr,qu->qwpr', t1_erisab, ya)
    t1_erisab = einsum('qwpr,sw->qspr', t1_erisab, yb)
    t1_erisab = t1_erisab.transpose(2, 3, 0, 1)
    return t1_erisaa, t1_erisab, t1_erisbb

def update_t1_fock_eris_uhf(mycc, imds, t1, eris=None):
    '''Compute the Fock matrix and ERIs dressed by T1 amplitudes.
    `t1_erisaa` and `t1_erisbb` are anti-symmetrized.
    '''
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    xa, xb, ya, yb = update_xy_uhf(mycc, t1)
    t1_focka, t1_fockb = update_fock_uhf(mycc, xa, xb, ya, yb, t1, eris)
    t1_erisaa, t1_erisab, t1_erisbb = update_eris_uhf(mycc, xa, xb, ya, yb, eris)
    imds.t1_fock = (t1_focka, t1_fockb)
    imds.t1_eris = (t1_erisaa, t1_erisab, t1_erisbb)
    return imds

def intermediates_t1t2_uhf(mycc, imds, t2):
    '''Intermediates for the T1 and T2 residual equation.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc

    t1_focka, t1_fockb = imds.t1_fock
    t1_erisaa, t1_erisab, t1_erisbb = imds.t1_eris
    t2aa, t2ab, t2bb = t2

    F_vv = t1_focka[nocca:, nocca:].copy()
    einsum('klcd,klbd->bc', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=F_vv, alpha=-0.5, beta=1.0)
    einsum('lkcd,lbkd->bc', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=F_vv, alpha=-1.0, beta=1.0)
    F_oo = t1_focka[:nocca, :nocca].copy()
    einsum('klcd,jlcd->kj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=F_oo, alpha=0.5, beta=1.0)
    einsum('kldc,jdlc->kj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=F_oo, alpha=1.0, beta=1.0)
    W_oooo = t1_erisaa[:nocca, :nocca, :nocca, :nocca].copy()
    einsum('klcd,ijcd->klij', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_oooo, alpha=0.5, beta=1.0)
    W_ovvo = t1_erisaa[:nocca, nocca:, nocca:, :nocca].copy()
    einsum('klcd,jlbd->kbcj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_ovvo, alpha=0.5, beta=1.0)
    einsum('klcd,jbld->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_ovvo, alpha=0.5, beta=1.0)
    W_OvVo = t1_erisab[nocca:, :noccb, :nocca, noccb:].transpose(1, 0, 3, 2).copy()
    einsum('klcd,jbld->kbcj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2ab, out=W_OvVo, alpha=0.5, beta=1.0)
    einsum('lkdc,jlbd->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2aa, out=W_OvVo, alpha=0.5, beta=1.0)

    F_VV = t1_fockb[noccb:, noccb:].copy()
    einsum('klcd,klbd->bc', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=F_VV, alpha=-0.5, beta=1.0)
    einsum('kldc,kdlb->bc', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=F_VV, alpha=-1.0, beta=1.0)
    F_OO = t1_fockb[:noccb, :noccb].copy()
    einsum('klcd,jlcd->kj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=F_OO, alpha=0.5, beta=1.0)
    einsum('lkcd,lcjd->kj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=F_OO, alpha=1.0, beta=1.0)
    W_OOOO = t1_erisbb[:noccb, :noccb, :noccb, :noccb].copy()
    einsum('klcd,ijcd->klij', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_OOOO, alpha=0.5, beta=1.0)
    W_OVVO = t1_erisbb[:noccb, noccb:, noccb:, :noccb].copy()
    einsum('klcd,jlbd->kbcj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_OVVO, alpha=0.5, beta=1.0)
    einsum('lkdc,ldjb->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_OVVO, alpha=0.5, beta=1.0)
    W_oVvO = t1_erisab[:nocca, noccb:, nocca:, :noccb].copy()
    einsum('klcd,ldjb->kbcj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2ab, out=W_oVvO, alpha=0.5, beta=1.0)
    einsum('klcd,jlbd->kbcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2bb, out=W_oVvO, alpha=0.5, beta=1.0)

    W_oOoO = t1_erisab[:nocca, :noccb, :nocca, :noccb].copy()
    einsum('klcd,icjd->klij', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_oOoO, alpha=1.0, beta=1.0)
    W_vOvO = - t1_erisab[nocca:, :noccb, nocca:, :noccb]
    einsum('lkcd,lajd->akcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vOvO, alpha=0.5, beta=1.0)
    W_VoVo = - t1_erisab[:nocca, noccb:, :nocca, noccb:].transpose(1, 0, 3, 2)
    einsum('kldc,idlb->bkci', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_VoVo, alpha=0.5, beta=1.0)
    W_vovo = - t1_erisaa[nocca:, :nocca, nocca:, :nocca]
    einsum('klcd,lida->akci', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_vovo, alpha=0.5, beta=1.0)
    einsum('klcd,iald->akci', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vovo, alpha=0.5, beta=1.0)
    W_VOVO = - t1_erisbb[noccb:, :noccb, noccb:, :noccb]
    einsum('klcd,ljdb->bkcj', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_VOVO, alpha=0.5, beta=1.0)
    einsum('lkdc,ldjb->bkcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_VOVO, alpha=0.5, beta=1.0)
    W_vOVo = t1_erisab[nocca:, :noccb, :nocca, noccb:].transpose(0, 1, 3, 2).copy()
    einsum('lkdc,ilad->akci', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2aa, out=W_vOVo, alpha=0.5, beta=1.0)
    einsum('lkdc,iald->akci', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2ab, out=W_vOVo, alpha=0.5, beta=1.0)
    W_VovO = t1_erisab[:nocca, noccb:, nocca:, :noccb].transpose(1, 0, 2, 3).copy()
    einsum('klcd,ljdb->bkcj', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2bb, out=W_VovO, alpha=0.5, beta=1.0)
    einsum('lkdc,ldjb->bkcj', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2ab, out=W_VovO, alpha=0.5, beta=1.0)
    imds.F_oo, imds.F_OO, imds.F_vv, imds.F_VV = F_oo, F_OO, F_vv, F_VV
    imds.W_oooo, imds.W_oOoO, imds.W_OOOO = W_oooo, W_oOoO, W_OOOO
    imds.W_ovvo, imds.W_oVvO, imds.W_OvVo, imds.W_OVVO = W_ovvo, W_oVvO, W_OvVo, W_OVVO,
    imds.W_vovo, imds.W_vOvO, imds.W_vOVo = W_vovo, W_vOvO, W_vOVo
    imds.W_VovO, imds.W_VoVo, imds.W_VOVO = W_VovO, W_VoVo, W_VOVO
    return imds

def compute_r1r2_uhf(mycc, imds, t2):
    '''Compute r1 and r2 without the contributions from T3 amplitudes.
    r2 will require a symmetry restoration step afterward.
    '''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    t1_focka, t1_fockb = imds.t1_fock
    t1_erisaa, t1_erisab, t1_erisbb = imds.t1_eris
    t2aa, t2ab, t2bb = t2

    F_oo, F_OO, F_vv, F_VV = imds.F_oo, imds.F_OO, imds.F_vv, imds.F_VV
    W_oooo, W_oOoO, W_OOOO = imds.W_oooo, imds.W_oOoO, imds.W_OOOO
    W_ovvo, W_oVvO, W_OvVo, W_OVVO = imds.W_ovvo, imds.W_oVvO, imds.W_OvVo, imds.W_OVVO
    W_vovo, W_vOvO, W_vOVo = imds.W_vovo, imds.W_vOvO, imds.W_vOVo
    W_VovO, W_VoVo, W_VOVO = imds.W_VovO, imds.W_VoVo, imds.W_VOVO

    r1a = t1_focka[nocca:, :nocca].T.copy()
    einsum('kc,ikac->ia', t1_focka[:nocca, nocca:], t2aa, out=r1a, alpha=1.0, beta=1.0)
    einsum('kc,iakc->ia', t1_fockb[:noccb, noccb:], t2ab, out=r1a, alpha=1.0, beta=1.0)
    einsum('akcd,ikcd->ia', t1_erisaa[nocca:, :nocca, nocca:, nocca:], t2aa, out=r1a, alpha=0.5, beta=1.0)
    einsum('akcd,ickd->ia', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=r1a, alpha=1.0, beta=1.0)
    einsum('klic,klac->ia', t1_erisaa[:nocca, :nocca, :nocca, nocca:], t2aa, out=r1a, alpha=-0.5, beta=1.0)
    einsum('klic,kalc->ia', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=r1a, alpha=-1.0, beta=1.0)

    r1b = t1_fockb[noccb:, :noccb].T.copy()
    einsum('kc,ikac->ia', t1_fockb[:noccb, noccb:], t2bb, out=r1b, alpha=1.0, beta=1.0)
    einsum('kc,kcia->ia', t1_focka[:nocca, nocca:], t2ab, out=r1b, alpha=1.0, beta=1.0)
    einsum('akcd,ikcd->ia', t1_erisbb[noccb:, :noccb, noccb:, noccb:], t2bb, out=r1b, alpha=0.5, beta=1.0)
    einsum('kadc,kdic->ia', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=r1b, alpha=1.0, beta=1.0)
    einsum('klic,klac->ia', t1_erisbb[:noccb, :noccb, :noccb, noccb:], t2bb, out=r1b, alpha=-0.5, beta=1.0)
    einsum('lkci,lcka->ia', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=r1b, alpha=-1.0, beta=1.0)

    r2aa = 0.25 * t1_erisaa[nocca:, nocca:, :nocca, :nocca].T
    einsum("bc,ijac->ijab", F_vv, t2aa, out=r2aa, alpha=0.5, beta=1.0)
    einsum("kj,ikab->ijab", F_oo, t2aa, out=r2aa, alpha=-0.5, beta=1.0)
    einsum("abcd,ijcd->ijab", t1_erisaa[nocca:, nocca:, nocca:, nocca:], t2aa, out=r2aa, alpha=0.125, beta=1.0)
    einsum("klij,klab->ijab", W_oooo, t2aa, out=r2aa, alpha=0.125, beta=1.0)
    einsum("kbcj,ikac->ijab", W_ovvo, t2aa, out=r2aa, alpha=1.0, beta=1.0)
    einsum("kbcj,iakc->ijab", W_OvVo, t2ab, out=r2aa, alpha=1.0, beta=1.0)
    W_ovvo = imds.W_ovvo = None
    W_OvVo = imds.W_OvVo = None

    r2ab = t1_erisab[nocca:, noccb:, :nocca, :noccb].transpose(2, 3, 0, 1).copy()
    r2ab = r2ab.transpose(0, 2, 1, 3)
    einsum("bc,iajc->iajb", F_VV, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("ac,icjb->iajb", F_vv, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("kj,iakb->iajb", F_OO, t2ab, out=r2ab, alpha=-1.0, beta=1.0)
    einsum("ki,kajb->iajb", F_oo, t2ab, out=r2ab, alpha=-1.0, beta=1.0)
    einsum("abcd,icjd->iajb", t1_erisab[nocca:, noccb:, nocca:, noccb:], t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("klij,kalb->iajb", W_oOoO, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("akcj,ickb->iajb", W_vOvO, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("akci,kcjb->iajb", W_vovo, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("akci,kjcb->iajb", W_vOVo, t2bb, out=r2ab, alpha=1.0, beta=1.0)
    einsum("bkcj,ikac->iajb", W_VovO, t2aa, out=r2ab, alpha=1.0, beta=1.0)
    einsum("bkcj,iakc->iajb", W_VOVO, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("bkci,kajc->iajb", W_VoVo, t2ab, out=r2ab, alpha=1.0, beta=1.0)
    W_vovo = imds.W_vovo = None
    W_vOvO = imds.W_vOvO = None
    W_vOVo = imds.W_vOVo = None
    W_VovO = imds.W_VovO = None
    W_VoVo = imds.W_VoVo = None
    W_VOVO = imds.W_VOVO = None

    r2bb = 0.25 * t1_erisbb[noccb:, noccb:, :noccb, :noccb].T
    einsum("bc,ijac->ijab", F_VV, t2bb, out=r2bb, alpha=0.5, beta=1.0)
    einsum("kj,ikab->ijab", F_OO, t2bb, out=r2bb, alpha=-0.5, beta=1.0)
    einsum("abcd,ijcd->ijab", t1_erisbb[noccb:, noccb:, noccb:, noccb:], t2bb, out=r2bb, alpha=0.125, beta=1.0)
    einsum("klij,klab->ijab", W_OOOO, t2bb, out=r2bb, alpha=0.125, beta=1.0)
    einsum("kbcj,ikac->ijab", W_OVVO, t2bb, out=r2bb, alpha=1.0, beta=1.0)
    einsum("kbcj,kcia->ijab", W_oVvO, t2ab, out=r2bb, alpha=1.0, beta=1.0)
    W_oVvO = imds.W_oVvO = None
    W_OVVO = imds.W_OVVO = None
    return [r1a, r1b], [r2aa, r2ab, r2bb]

def r1r2_add_t3_tri_uhf_(mycc, imds, r1, r2, t3):
    '''Add the T3 contributions to r1 and r2. T3 amplitudes are stored in triangular form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    t1_focka, t1_fockb = imds.t1_fock
    t1_erisaa, t1_erisab, t1_erisbb = imds.t1_eris

    t3aaa, t3aab, t3bba, t3bbb = t3
    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    for i0, i1 in lib.prange(0, nocca, blksize_o_aaa):
        bi = i1 - i0
        for m0, m1 in lib.prange(0, nocca, blksize_o_aaa):
            bm = m1 - m0
            for n0, n1 in lib.prange(0, nocca, blksize_o_aaa):
                bn = n1 - n0
                for a0, a1 in lib.prange(0, nvira, blksize_v_aaa):
                    ba = a1 - a0
                    for e0, e1 in lib.prange(0, nvira, blksize_v_aaa):
                        be = e1 - e0
                        for f0, f1 in lib.prange(0, nvira, blksize_v_aaa):
                            bf = f1 - f0
                            _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, m0, m1, n0, n1, a0, a1, e0, e1, f0, f1)
                            einsum('mnef,imnaef->ia',
                                t1_erisaa[m0:m1, n0:n1, nocca + e0:nocca + e1, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf], out=r1a[i0:i1, a0:a1], alpha=0.25, beta=1.0)
                            einsum("nf,imnaef->imae", t1_focka[n0:n1, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2aa[i0:i1, m0:m1, a0:a1, e0:e1], alpha=0.25, beta=1.0)
                            einsum("bnef,imnaef->imab",
                                t1_erisaa[nocca:, n0:n1, nocca + e0:nocca + e1, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2aa[i0:i1, m0:m1, a0:a1, :], alpha=0.25, beta=1.0)
                            einsum("mnjf,imnaef->ijae", t1_erisaa[m0:m1, n0:n1, :nocca, nocca + f0:nocca + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2aa[i0:i1, :, a0:a1, e0:e1], alpha=-0.25, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aab.dtype)
    for i0, i1 in lib.prange(0, nocca, blksize_o_aab):
        bi = i1 - i0
        for n0, n1 in lib.prange(0, nocca, blksize_o_aab):
            bn = n1 - n0
            for a0, a1 in lib.prange(0, nvira, blksize_v_aab):
                ba = a1 - a0
                for f0, f1 in lib.prange(0, nvira, blksize_v_aab):
                    bf = f1 - f0
                    _unp_aab_(mycc, t3aab, t3_tmp, i0, i1, n0, n1, a0, a1, f0, f1)
                    einsum('nmfe,inafme->ia', t1_erisab[n0:n1, :noccb, nocca + f0:nocca + f1, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r1a[i0:i1, a0:a1], alpha=1.0, beta=1.0)
                    einsum('inaf,inafme->me',
                        t1_erisaa[i0:i1, n0:n1, nocca + a0:nocca + a1, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r1b, alpha=0.25, beta=1.0)
                    einsum("me,inafme->inaf", t1_fockb[:noccb, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2aa[i0:i1, n0:n1, a0:a1, f0:f1], alpha=0.25, beta=1.0)
                    einsum("emfb,inafmb->inae", t1_erisab[nocca:, :noccb, nocca + f0:nocca + f1, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2aa[i0:i1, n0:n1, a0:a1, :], alpha=0.5, beta=1.0)
                    einsum("njme,inafje->imaf", t1_erisab[n0:n1, :noccb, :nocca, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2aa[i0:i1, :, a0:a1, f0:f1], alpha=-0.5, beta=1.0)
                    einsum("nf,inafjb->iajb", t1_focka[n0:n1, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, a0:a1, :, :], alpha=1.0, beta=1.0)
                    einsum("nbfe,inafje->iajb", t1_erisab[n0:n1, noccb:, nocca + f0:nocca + f1, noccb:],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, a0:a1, :, :], alpha=1.0, beta=1.0)
                    einsum("enaf,inafjb->iejb",
                        t1_erisaa[nocca:, n0:n1, nocca + a0:nocca + a1, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, ...], alpha=0.5, beta=1.0)
                    einsum("nmfj,inafmb->iajb", t1_erisab[n0:n1, :noccb, nocca + f0:nocca + f1, :noccb],
                        t3_tmp[:bi, :bn, :ba, :bf], out=r2ab[i0:i1, a0:a1, :, :], alpha=-1.0, beta=1.0)
                    einsum("inmf,inafjb->majb", t1_erisaa[i0:i1, n0:n1, :nocca, nocca + f0:nocca + f1],
                        t3_tmp[:bi, :bn, :ba, :bf, :, :], out=r2ab[:, a0:a1, :, :], alpha=-0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bba.dtype)
    for m0, m1 in lib.prange(0, noccb, blksize_o_aab):
        bm = m1 - m0
        for n0, n1 in lib.prange(0, noccb, blksize_o_aab):
            bn = n1 - n0
            for e0, e1 in lib.prange(0, nvirb, blksize_v_aab):
                be = e1 - e0
                for f0, f1 in lib.prange(0, nvirb, blksize_v_aab):
                    bf = f1 - f0
                    _unp_bba_(mycc, t3bba, t3_tmp, m0, m1, n0, n1, e0, e1, f0, f1)
                    einsum('mnef,mnefia->ia',
                        t1_erisbb[m0:m1, n0:n1, noccb + e0:noccb + e1, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r1a, alpha=0.25, beta=1.0)
                    einsum('inaf,mnefia->me', t1_erisab[:nocca, n0:n1, nocca:, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r1b[m0:m1, e0:e1], alpha=1.0, beta=1.0)
                    einsum("nf,mnefia->iame", t1_fockb[n0:n1, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, e0:e1], alpha=1.0, beta=1.0)
                    einsum("bnef,mnefia->iamb",
                        t1_erisbb[noccb:, n0:n1, noccb + e0:noccb + e1, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, :], alpha=0.5, beta=1.0)
                    einsum("anbf,mnefib->iame", t1_erisab[nocca:, n0:n1, nocca:, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, e0:e1], alpha=1.0, beta=1.0)
                    einsum("mnjf,mnefia->iaje", t1_erisbb[m0:m1, n0:n1, :noccb, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, :, e0:e1], alpha=-0.5, beta=1.0)
                    einsum("jnif,mnefja->iame", t1_erisab[:nocca, n0:n1, :nocca, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2ab[:, :, m0:m1, e0:e1], alpha=-1.0, beta=1.0)
                    einsum("ia,mnefia->mnef", t1_focka[:nocca, nocca:],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2bb[m0:m1, n0:n1, e0:e1, f0:f1], alpha=0.25, beta=1.0)
                    einsum("iabf,mnefib->mnea", t1_erisab[:nocca, noccb:, nocca:, noccb + f0:noccb + f1],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2bb[m0:m1, n0:n1, e0:e1, :], alpha=0.5, beta=1.0)
                    einsum("jnai,mnefja->mief", t1_erisab[:nocca, n0:n1, nocca:, :noccb],
                        t3_tmp[:bm, :bn, :be, :bf], out=r2bb[m0:m1, :, e0:e1, f0:f1], alpha=-0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    for i0, i1 in lib.prange(0, noccb, blksize_o_aaa):
        bi = i1 - i0
        for m0, m1 in lib.prange(0, noccb, blksize_o_aaa):
            bm = m1 - m0
            for n0, n1 in lib.prange(0, noccb, blksize_o_aaa):
                bn = n1 - n0
                for a0, a1 in lib.prange(0, nvirb, blksize_v_aaa):
                    ba = a1 - a0
                    for e0, e1 in lib.prange(0, nvirb, blksize_v_aaa):
                        be = e1 - e0
                        for f0, f1 in lib.prange(0, nvirb, blksize_v_aaa):
                            bf = f1 - f0
                            _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, m0, m1, n0, n1, a0, a1, e0, e1, f0, f1)
                            einsum('mnef,imnaef->ia',
                                t1_erisbb[m0:m1, n0:n1, noccb + e0 : noccb + e1, noccb + f0: noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf], out=r1b[i0:i1, a0:a1], alpha=0.25, beta=1.0)
                            einsum("nf,imnaef->imae", t1_fockb[n0:n1, noccb + f0:noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2bb[i0:i1, m0:m1, a0:a1, e0:e1], alpha=0.25, beta=1.0)
                            einsum("bnef,imnaef->imab",
                                t1_erisbb[noccb:, n0:n1, noccb + e0:noccb + e1, noccb + f0:noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2bb[i0:i1, m0:m1, a0:a1, :], alpha=0.25, beta=1.0)
                            einsum("mnjf,imnaef->ijae", t1_erisbb[m0:m1, n0:n1, :noccb, noccb + f0:noccb + f1],
                                t3_tmp[:bi, :bm, :bn, :ba, :be, :bf],
                                out=r2bb[i0:i1, :, a0:a1, e0:e1], alpha=-0.25, beta=1.0)
    t3_tmp = None
    return r1, r2

def antisymmetrize_r2_uhf_(r2):
    r2aa, r2ab, r2bb = r2
    r2aa -= r2aa.transpose(1, 0, 2, 3)
    r2aa -= r2aa.transpose(0, 1, 3, 2)
    r2bb -= r2bb.transpose(1, 0, 2, 3)
    r2bb -= r2bb.transpose(0, 1, 3, 2)
    return r2

def r1r2_divide_e_uhf_(mycc, r1, r2, mo_energy):
    r1a, r1b = r1
    r2aa, r2ab, r2bb = r2
    nocca, noccb = r1a.shape[0], r1b.shape[0]

    eia_a = mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:] - mycc.level_shift
    r1a /= eia_a
    eia_b = mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:] - mycc.level_shift
    r1b /= eia_b

    eijab_aa = eia_a[:, None, :, None] + eia_a[None, :, None, :]
    r2aa /= eijab_aa
    eijab_ab = eia_a[:, :, None, None] + eia_b[None, None, :, :]
    r2ab /= eijab_ab
    eijab_bb = eia_b[:, None, :, None] + eia_b[None, :, None, :]
    r2bb /= eijab_bb
    eia_a, eia_b, eijab_aa, eijab_ab, eijab_bb = None, None, None, None, None
    return r1, r2

def intermediates_t3_uhf(mycc, imds, t2):
    '''Intermediates for the T3 residual equation (excluding T3 contributions).'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    t1_focka, t1_fockb = imds.t1_fock
    t1_erisaa, t1_erisab, t1_erisbb = imds.t1_eris
    t2aa, t2ab, t2bb = t2

    W_vvvv = t1_erisaa[nocca:, nocca:, nocca:, nocca:].copy()
    einsum('lmde,lmab->abde', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_vvvv, alpha=0.5, beta=1.0)
    W_voov = t1_erisaa[nocca:, :nocca, :nocca, nocca:].copy()
    einsum('mled,imae->alid', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2aa, out=W_voov, alpha=1.0, beta=1.0)
    einsum('lmde,iame->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_voov, alpha=1.0, beta=1.0)
    W_vOoV = t1_erisab[nocca:, :noccb, :nocca, noccb:].copy()
    einsum('mled,imae->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2aa, out=W_vOoV, alpha=1.0, beta=1.0)
    einsum('mled,iame->alid', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2ab, out=W_vOoV, alpha=1.0, beta=1.0)
    W_vvvo = t1_erisaa[nocca:, nocca:, nocca:, :nocca].copy()
    einsum('lbed,klce->bcdk', t1_erisaa[:nocca, nocca:, nocca:, nocca:], t2aa, out=W_vvvo, alpha=2.0, beta=1.0)
    einsum('blde,kcle->bcdk', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=W_vvvo, alpha=2.0, beta=1.0)
    einsum('lmdk,lmbc->bcdk', t1_erisaa[:nocca, :nocca, nocca:, :nocca], t2aa, out=W_vvvo, alpha=0.5, beta=1.0)
    W_ovoo = t1_erisaa[:nocca, nocca:, :nocca, :nocca].copy()
    einsum('ld,jkdc->lcjk', t1_focka[:nocca, nocca:], t2aa, out=W_ovoo, alpha=1.0, beta=1.0)
    einsum('mldj,kmcd->lcjk', t1_erisaa[:nocca, :nocca, nocca:, :nocca], t2aa, out=W_ovoo, alpha=2.0, beta=1.0)
    einsum('lmjd,kcmd->lcjk', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=W_ovoo, alpha=2.0, beta=1.0)
    einsum('lcde,jkde->lcjk', t1_erisaa[:nocca, nocca:, nocca:, nocca:], t2aa, out=W_ovoo, alpha=0.5, beta=1.0)

    W_VVVV = t1_erisbb[noccb:, noccb:, noccb:, noccb:].copy()
    einsum('lmde,lmab->abde', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_VVVV, alpha=0.5, beta=1.0)
    W_VOOV = t1_erisbb[noccb:, :noccb, :noccb, noccb:].copy()
    einsum('mled,imae->alid', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t2bb, out=W_VOOV, alpha=1.0, beta=1.0)
    einsum('mled,meia->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_VOOV, alpha=1.0, beta=1.0)
    W_VoOv = t1_erisab[:nocca, noccb:, nocca:, :noccb].transpose(1, 0, 3, 2).copy()
    einsum('lmde,imae->alid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2bb, out=W_VoOv, alpha=1.0, beta=1.0)
    einsum('mled,meia->alid', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t2ab, out=W_VoOv, alpha=1.0, beta=1.0)
    W_VVVO = t1_erisbb[noccb:, noccb:, noccb:, :noccb].copy()
    einsum('lbed,klce->bcdk', t1_erisbb[:noccb, noccb:, noccb:, noccb:], t2bb, out=W_VVVO, alpha=2.0, beta=1.0)
    einsum('lbed,lekc->bcdk', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=W_VVVO, alpha=2.0, beta=1.0)
    einsum('lmdk,lmbc->bcdk', t1_erisbb[:noccb, :noccb, noccb:, :noccb], t2bb, out=W_VVVO, alpha=0.5, beta=1.0)
    W_OVOO = t1_erisbb[:noccb, noccb:, :noccb, :noccb].copy()
    einsum('ld,jkdc->lcjk', t1_fockb[:noccb, noccb:], t2bb, out=W_OVOO, alpha=1.0, beta=1.0)
    einsum('mldj,kmcd->lcjk', t1_erisbb[:noccb, :noccb, noccb:, :noccb], t2bb, out=W_OVOO, alpha=2.0, beta=1.0)
    einsum('mldj,mdkc->lcjk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=W_OVOO, alpha=2.0, beta=1.0)
    einsum('lcde,jkde->lcjk', t1_erisbb[:noccb, noccb:, noccb:, noccb:], t2bb, out=W_OVOO, alpha=0.5, beta=1.0)

    W_vVvV = t1_erisab[nocca:, noccb:, nocca:, noccb:].copy()
    einsum('lmed,lbmc->bced', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vVvV, alpha=1.0, beta=1.0)
    W_oVoV = t1_erisab[:nocca, noccb:, :nocca, noccb:].copy()
    einsum('lmed,iemc->lcid', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_oVoV, alpha=-1.0, beta=1.0)
    W_vOvO = t1_erisab[nocca:, :noccb, nocca:, :noccb].copy()
    einsum('mlde,make->aldk', t1_erisab[:nocca, :noccb, nocca:, noccb:], t2ab, out=W_vOvO, alpha=-1.0, beta=1.0)
    W_vVvO = t1_erisab[nocca:, noccb:, nocca:, :noccb].copy()
    einsum('lbed,lekc->bcdk', t1_erisaa[:nocca, nocca:, nocca:, nocca:], t2ab, out=W_vVvO, alpha=1.0, beta=1.0)
    einsum('blde,lkec->bcdk', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2bb, out=W_vVvO, alpha=1.0, beta=1.0)
    einsum('lcde,lbke->bcdk', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=W_vVvO, alpha=-1.0, beta=1.0)
    einsum('lmdk,lbmc->bcdk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=W_vVvO, alpha=1.0, beta=1.0)
    W_oVoO = t1_erisab[:nocca, noccb:, :nocca, :noccb].copy()
    einsum('ld,jdkc->lcjk', t1_focka[:nocca, nocca:], t2ab, out=W_oVoO, alpha=1.0, beta=1.0)
    einsum('mldj,mdkc->lcjk', t1_erisaa[:nocca, :nocca, nocca:, :nocca], t2ab, out=W_oVoO, alpha=1.0, beta=1.0)
    einsum('lmjd,mkdc->lcjk', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2bb, out=W_oVoO, alpha=1.0, beta=1.0)
    einsum('lmdk,jdmc->lcjk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2ab, out=W_oVoO, alpha=-1.0, beta=1.0)
    einsum('lcde,jdke->lcjk', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2ab, out=W_oVoO, alpha=1.0, beta=1.0)
    W_vVoV = t1_erisab[nocca:, noccb:, :nocca, noccb:].copy()
    einsum('bled,jelc->bcjd', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=W_vVoV, alpha=-1.0, beta=1.0)
    einsum('lced,jlbe->bcjd', t1_erisab[:nocca, noccb:, nocca:, noccb:], t2aa, out=W_vVoV, alpha=1.0, beta=1.0)
    einsum('lced,jble->bcjd', t1_erisbb[:noccb, noccb:, noccb:, noccb:], t2ab, out=W_vVoV, alpha=1.0, beta=1.0)
    einsum('mljd,mblc->bcjd', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=W_vVoV, alpha=1.0, beta=1.0)
    W_vOoO = t1_erisab[nocca:, :noccb, :nocca, :noccb].copy()
    einsum('ld,jakd->aljk', t1_fockb[:noccb, noccb:], t2ab, out=W_vOoO, alpha=1.0, beta=1.0)
    einsum('mljd,makd->aljk', t1_erisab[:nocca, :noccb, :nocca, noccb:], t2ab, out=W_vOoO, alpha=-1.0, beta=1.0)
    einsum('mldk,jmad->aljk', t1_erisab[:nocca, :noccb, nocca:, :noccb], t2aa, out=W_vOoO, alpha=1.0, beta=1.0)
    einsum('mldk,jamd->aljk', t1_erisbb[:noccb, :noccb, noccb:, :noccb], t2ab, out=W_vOoO, alpha=1.0, beta=1.0)
    einsum('alde,jdke->aljk', t1_erisab[nocca:, :noccb, nocca:, noccb:], t2ab, out=W_vOoO, alpha=1.0, beta=1.0)
    imds.W_ovoo, imds.W_oVoO, imds.W_OVOO = W_ovoo, W_oVoO, W_OVOO
    imds.W_vOoO, imds.W_vVoV = W_vOoO, W_vVoV
    imds.W_voov, imds.W_vOoV, imds.W_VoOv, imds.W_VOOV = W_voov, W_vOoV, W_VoOv, W_VOOV
    imds.W_oVoV, imds.W_vOvO = W_oVoV, W_vOvO
    imds.W_vvvo, imds.W_vVvO, imds.W_VVVO = W_vvvo, W_vVvO, W_VVVO
    imds.W_vvvv, imds.W_vVvV, imds.W_VVVV = W_vvvv, W_vVvV, W_VVVV
    return imds

def intermediates_t3_add_t3_tri_uhf(mycc, imds, t3):
    '''Add the T3-dependent contributions to the T3 intermediates, with T3 stored in triangular form.'''
    '''Update W_ovoo, W_oVoO, W_OVOO, W_vOoO, W_vvvo, W_vVvO, W_VVVO, W_vVoV inplace.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    t1_erisaa, t1_erisab, t1_erisbb = imds.t1_eris
    t3aaa, t3aab, t3bba, t3bbb = t3

    W_ovoo, W_oVoO, W_OVOO = imds.W_ovoo, imds.W_oVoO, imds.W_OVOO
    W_vvvo, W_vVvO, W_VVVO = imds.W_vvvo, imds.W_vVvO, imds.W_VVVO
    W_vOoO, W_vVoV = imds.W_vOoO, imds.W_vVoV

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    for l0, l1 in lib.prange(0, nocca, blksize_o_aaa):
        bl = l1 - l0
        for m0, m1 in lib.prange(0, nocca, blksize_o_aaa):
            bm = m1 - m0
            for k0, k1 in lib.prange(0, nocca, blksize_o_aaa):
                bk = k1 - k0
                for b0, b1 in lib.prange(0, nvira, blksize_v_aaa):
                    bb = b1 - b0
                    for e0, e1 in lib.prange(0, nvira, blksize_v_aaa):
                        be = e1 - e0
                        for c0, c1 in lib.prange(0, nvira, blksize_v_aaa):
                            bc = c1 - c0
                            _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, k0, k1, b0, b1, e0, e1, c0, c1)
                            einsum('lmde,lmkbec->bcdk', t1_erisaa[l0:l1, m0:m1, nocca:, nocca + e0:nocca + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=W_vvvo[b0:b1, c0:c1, :, k0:k1], alpha=-0.5, beta=1.0)
                            einsum('jmbe,lmkbec->jclk',
                                    t1_erisaa[:nocca, m0:m1, nocca + b0:nocca + b1, nocca + e0:nocca + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=W_ovoo[:, c0:c1, l0:l1, k0:k1], alpha=0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    for l0, l1 in lib.prange(0, noccb, blksize_o_aaa):
        bl = l1 - l0
        for m0, m1 in lib.prange(0, noccb, blksize_o_aaa):
            bm = m1 - m0
            for k0, k1 in lib.prange(0, noccb, blksize_o_aaa):
                bk = k1 - k0
                for b0, b1 in lib.prange(0, nvirb, blksize_v_aaa):
                    bb = b1 - b0
                    for e0, e1 in lib.prange(0, nvirb, blksize_v_aaa):
                        be = e1 - e0
                        for c0, c1 in lib.prange(0, nvirb, blksize_v_aaa):
                            bc = c1 - c0
                            _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, k0, k1, b0, b1, e0, e1, c0, c1)
                            einsum('lmde,lmkbec->bcdk', t1_erisbb[l0:l1, m0:m1, noccb:, noccb + e0:noccb + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=W_VVVO[b0:b1, c0:c1, :, k0:k1], alpha=-0.5, beta=1.0)
                            einsum('jmbe,lmkbec->jclk',
                                    t1_erisbb[:noccb, m0:m1, noccb + b0:noccb + b1, noccb + e0:noccb + e1],
                                    t3_tmp[:bl, :bm, :bk, :bb, :be, :bc],
                                    out=W_OVOO[:, c0:c1, l0:l1, k0:k1], alpha=0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aab.dtype)
    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, nocca, blksize_o_aab):
            bk = k1 - k0
            for b0, b1 in lib.prange(0, nvira, blksize_v_aab):
                bb = b1 - b0
                for c0, c1 in lib.prange(0, nvira, blksize_v_aab):
                    bc = c1 - c0
                    _unp_aab_(mycc, t3aab, t3_tmp, l0, l1, k0, k1, b0, b1, c0, c1)
                    einsum('lmde,lkbcme->bcdk', t1_erisab[l0:l1, :noccb, nocca:, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_vvvo[b0:b1, c0:c1, :, k0:k1], alpha=-1.0, beta=1.0)
                    einsum('jmbe,lkbcme->jclk', t1_erisab[:nocca, :noccb, nocca + b0 : nocca + b1, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_ovoo[:, c0:c1, l0:l1, k0:k1], alpha=1.0, beta=1.0)
                    einsum('lkdc,lkbcme->bedm', t1_erisaa[l0:l1, k0:k1, nocca:, nocca + c0: nocca + c1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_vVvO[b0:b1, :, :, :], alpha=-0.5, beta=1.0)
                    einsum('jkbc,lkbcme->jelm',
                        t1_erisaa[:nocca, k0:k1, nocca + b0:nocca + b1, nocca + c0:nocca + c1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_oVoO[:, :, l0:l1, :], alpha=0.5, beta=1.0)
                    einsum('kmcd,lkbcme->beld', t1_erisab[k0:k1, :noccb, nocca + c0: nocca + c1, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_vVoV[b0:b1, :, l0:l1, :], alpha=-1.0, beta=1.0)
                    einsum('kjce,lkbcme->bjlm', t1_erisab[k0:k1, :noccb, nocca + c0:nocca + c1, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_vOoO[b0:b1, :, l0:l1, :], alpha=1.0, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bba.dtype)
    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
        bl = l1 - l0
        for k0, k1 in lib.prange(0, noccb, blksize_o_aab):
            bk = k1 - k0
            for b0, b1 in lib.prange(0, nvirb, blksize_v_aab):
                bb = b1 - b0
                for c0, c1 in lib.prange(0, nvirb, blksize_v_aab):
                    bc = c1 - c0
                    _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, k0, k1, b0, b1, c0, c1)
                    einsum('mled,lkbcme->bcdk', t1_erisab[:nocca, l0:l1, nocca:, noccb:],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_VVVO[b0:b1, c0:c1, :, k0:k1], alpha=-1.0, beta=1.0)
                    einsum('mjeb,lkbcme->jclk', t1_erisab[:nocca, :noccb, nocca:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_OVOO[:, c0:c1, l0:l1, k0:k1], alpha=1.0, beta=1.0)
                    einsum('mldb,lkbcme->ecdk', t1_erisab[:nocca, l0:l1, nocca:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_vVvO[:, c0:c1, :, k0:k1], alpha=-1.0, beta=1.0)
                    einsum('jleb,lkbcme->jcmk', t1_erisab[:nocca, l0:l1, nocca:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_oVoO[:, c0:c1, :, k0:k1], alpha=1.0, beta=1.0)
                    einsum('kldb,lkbcme->ecmd', t1_erisbb[k0:k1, l0:l1, noccb:, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_vVoV[:, c0:c1, :, :], alpha=-0.5, beta=1.0)
                    einsum('jlcb,lkbcme->ejmk',
                        t1_erisbb[:noccb, l0:l1, noccb + c0:noccb + c1, noccb + b0:noccb + b1],
                        t3_tmp[:bl, :bk, :bb, :bc], out=W_vOoO[:, :, :, k0:k1], alpha=0.5, beta=1.0)
    t3_tmp = None
    return imds

def compute_r3aaa_tri_uhf(mycc, imds, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    t2aa = t2[0]
    t3aaa, t3aab = t3[0:2]

    F_oo, F_vv = imds.F_oo, imds.F_vv
    W_oooo, W_ovoo, W_vvvo, W_vvvv = imds.W_oooo, imds.W_ovoo, imds.W_vvvo, imds.W_vvvv
    W_voov, W_vOoV = imds.W_voov, imds.W_vOoV

    r3aaa = np.zeros_like(t3aaa)

    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3aaa.dtype)
    t3_tmp_2 = np.empty((blksize_o_aaa,) * 2 + (blksize_v_aaa,) * 2 + (noccb,) + (nvirb,), dtype=t3aaa.dtype)
    for k0, k1 in lib.prange(0, nocca, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvira, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            bijkabc = (slice(None, bi), slice(None, bj), slice(None, bk),
                                        slice(None, ba), slice(None, bb), slice(None, bc))

                            einsum("bcdk,ijad->ijkabc", W_vvvo[b0:b1, c0:c1, :, k0:k1],
                                t2aa[i0:i1, j0:j1, a0:a1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=0.0)
                            einsum("cbdk,ijad->ijkabc", W_vvvo[c0:c1, b0:b1, :, k0:k1],
                                t2aa[i0:i1, j0:j1, a0:a1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("acdk,ijbd->ijkabc", W_vvvo[a0:a1, c0:c1, :, k0:k1],
                                t2aa[i0:i1, j0:j1, b0:b1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("abdk,ijcd->ijkabc", W_vvvo[a0:a1, b0:b1, :, k0:k1],
                                t2aa[i0:i1, j0:j1, c0:c1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("cadk,ijbd->ijkabc", W_vvvo[c0:c1, a0:a1, :, k0:k1],
                                t2aa[i0:i1, j0:j1, b0:b1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("badk,ijcd->ijkabc", W_vvvo[b0:b1, a0:a1, :, k0:k1],
                                t2aa[i0:i1, j0:j1, c0:c1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("bcdj,ikad->ijkabc", W_vvvo[b0:b1, c0:c1, :, j0:j1],
                                t2aa[i0:i1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("cbdj,ikad->ijkabc", W_vvvo[c0:c1, b0:b1, :, j0:j1],
                                t2aa[i0:i1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("acdj,ikbd->ijkabc", W_vvvo[a0:a1, c0:c1, :, j0:j1],
                                t2aa[i0:i1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("abdj,ikcd->ijkabc", W_vvvo[a0:a1, b0:b1, :, j0:j1],
                                t2aa[i0:i1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("cadj,ikbd->ijkabc", W_vvvo[c0:c1, a0:a1, :, j0:j1],
                                t2aa[i0:i1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("badj,ikcd->ijkabc", W_vvvo[b0:b1, a0:a1, :, j0:j1],
                                t2aa[i0:i1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("bcdi,jkad->ijkabc", W_vvvo[b0:b1, c0:c1, :, i0:i1],
                                t2aa[j0:j1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("cbdi,jkad->ijkabc", W_vvvo[c0:c1, b0:b1, :, i0:i1],
                                t2aa[j0:j1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("acdi,jkbd->ijkabc", W_vvvo[a0:a1, c0:c1, :, i0:i1],
                                t2aa[j0:j1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("abdi,jkcd->ijkabc", W_vvvo[a0:a1, b0:b1, :, i0:i1],
                                t2aa[j0:j1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("cadi,jkbd->ijkabc", W_vvvo[c0:c1, a0:a1, :, i0:i1],
                                t2aa[j0:j1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("badi,jkcd->ijkabc", W_vvvo[b0:b1, a0:a1, :, i0:i1],
                                t2aa[j0:j1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)

                            einsum("lcjk,ilab->ijkabc", W_ovoo[:, c0:c1, j0:j1, k0:k1],
                                t2aa[i0:i1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lbjk,ilac->ijkabc", W_ovoo[:, b0:b1, j0:j1, k0:k1],
                                t2aa[i0:i1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lajk,ilbc->ijkabc", W_ovoo[:, a0:a1, j0:j1, k0:k1],
                                t2aa[i0:i1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lckj,ilab->ijkabc", W_ovoo[:, c0:c1, k0:k1, j0:j1],
                                t2aa[i0:i1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lbkj,ilac->ijkabc", W_ovoo[:, b0:b1, k0:k1, j0:j1],
                                t2aa[i0:i1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lakj,ilbc->ijkabc", W_ovoo[:, a0:a1, k0:k1, j0:j1],
                                t2aa[i0:i1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lcik,jlab->ijkabc", W_ovoo[:, c0:c1, i0:i1, k0:k1],
                                t2aa[j0:j1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lbik,jlac->ijkabc", W_ovoo[:, b0:b1, i0:i1, k0:k1],
                                t2aa[j0:j1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("laik,jlbc->ijkabc", W_ovoo[:, a0:a1, i0:i1, k0:k1],
                                t2aa[j0:j1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lcij,klab->ijkabc", W_ovoo[:, c0:c1, i0:i1, j0:j1],
                                t2aa[k0:k1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lbij,klac->ijkabc", W_ovoo[:, b0:b1, i0:i1, j0:j1],
                                t2aa[k0:k1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("laij,klbc->ijkabc", W_ovoo[:, a0:a1, i0:i1, j0:j1],
                                t2aa[k0:k1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lcki,jlab->ijkabc", W_ovoo[:, c0:c1, k0:k1, i0:i1],
                                t2aa[j0:j1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lbki,jlac->ijkabc", W_ovoo[:, b0:b1, k0:k1, i0:i1],
                                t2aa[j0:j1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("laki,jlbc->ijkabc", W_ovoo[:, a0:a1, k0:k1, i0:i1],
                                t2aa[j0:j1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lcji,klab->ijkabc", W_ovoo[:, c0:c1, j0:j1, i0:i1],
                                t2aa[k0:k1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lbji,klac->ijkabc", W_ovoo[:, b0:b1, j0:j1, i0:i1],
                                t2aa[k0:k1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("laji,klbc->ijkabc", W_ovoo[:, a0:a1, j0:j1, i0:i1],
                                t2aa[k0:k1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)

                            for d0, d1 in lib.prange(0, nvira, blksize_v_aaa):
                                bd = d1 - d0
                                _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, d0, d1)
                                einsum("cd,ijkabd->ijkabc", F_vv[c0:c1, d0:d1],
                                    t3_tmp[:bi, :bj, :bk, :ba, :bb, :bd], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, c0, c1, d0, d1)
                                einsum("bd,ijkacd->ijkabc", F_vv[b0:b1, d0:d1],
                                    t3_tmp[:bi, :bj, :bk, :ba, :bc, :bd], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, b0, b1, c0, c1, d0, d1)
                                einsum("ad,ijkbcd->ijkabc", F_vv[a0:a1, d0:d1],
                                    t3_tmp[:bi, :bj, :bk, :bb, :bc, :bd], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                for e0, e1 in lib.prange(0, nvira, blksize_v_aaa):
                                    be = e1 - e0
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, c0, c1)
                                    einsum("abde,ijkdec->ijkabc", W_vvvv[a0:a1, b0:b1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bc], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, b0, b1)
                                    einsum("acde,ijkdeb->ijkabc", W_vvvv[a0:a1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bb], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, a0, a1)
                                    einsum("bcde,ijkdea->ijkabc", W_vvvv[b0:b1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :ba], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)

                            for l0, l1 in lib.prange(0, nocca, blksize_o_aaa):
                                bl = l1 - l0
                                _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, j0, j1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum("lk,ijlabc->ijkabc", F_oo[l0:l1, k0:k1],
                                    t3_tmp[:bi, :bj, :bl, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                _unp_aaa_(mycc, t3aaa, t3_tmp, i0, i1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum("lj,iklabc->ijkabc", F_oo[l0:l1, j0:j1],
                                    t3_tmp[:bi, :bk, :bl, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                _unp_aaa_(mycc, t3aaa, t3_tmp, j0, j1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum("li,jklabc->ijkabc", F_oo[l0:l1, i0:i1],
                                    t3_tmp[:bj, :bk, :bl, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                for m0, m1 in lib.prange(0, nocca, blksize_o_aaa):
                                    bm = m1 - m0
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, k0, k1, a0, a1, b0, b1, c0, c1)
                                    einsum("lmij,lmkabc->ijkabc", W_oooo[l0:l1, m0:m1, i0:i1, j0:j1],
                                        t3_tmp[:bl, :bm, :bk, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, j0, j1, a0, a1, b0, b1, c0, c1)
                                    einsum("lmik,lmjabc->ijkabc", W_oooo[l0:l1, m0:m1, i0:i1, k0:k1],
                                        t3_tmp[:bl, :bm, :bj, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, m0, m1, i0, i1, a0, a1, b0, b1, c0, c1)
                                    einsum("lmjk,lmiabc->ijkabc", W_oooo[l0:l1, m0:m1, j0:j1, k0:k1],
                                        t3_tmp[:bl, :bm, :bi, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)

                            for l0, l1 in lib.prange(0, nocca, blksize_o_aaa):
                                bl = l1 - l0
                                for d0, d1 in lib.prange(0, nvira, blksize_v_aaa):
                                    bd = d1 - d0
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum("alid,ljkdbc->ijkabc", W_voov[a0:a1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum("blid,ljkdac->ijkabc", W_voov[b0:b1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum("clid,ljkdab->ijkabc", W_voov[c0:c1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bb], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum("aljd,likdbc->ijkabc", W_voov[a0:a1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :bb, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum("bljd,likdac->ijkabc", W_voov[b0:b1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum("cljd,likdab->ijkabc", W_voov[c0:c1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bb], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, b0, b1, c0, c1)
                                    einsum("alkd,lijdbc->ijkabc", W_voov[a0:a1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, c0, c1)
                                    einsum("blkd,lijdac->ijkabc", W_voov[b0:b1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_aaa_(mycc, t3aaa, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, b0, b1)
                                    einsum("clkd,lijdab->ijkabc", W_voov[c0:c1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bb], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)

                            _unp_aab_(mycc, t3aab, t3_tmp_2, j0, j1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("alid,jkbcld->ijkabc", W_vOoV[a0:a1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :bb, :bc, :, :], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, j0, j1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("blid,jkacld->ijkabc", W_vOoV[b0:b1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bc, :, :], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, j0, j1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("clid,jkabld->ijkabc", W_vOoV[c0:c1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bb, :, :], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, i0, i1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("aljd,ikbcld->ijkabc", W_vOoV[a0:a1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :bb, :bc, :, :], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, i0, i1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("bljd,ikacld->ijkabc", W_vOoV[b0:b1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bc, :, :], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, i0, i1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("cljd,ikabld->ijkabc", W_vOoV[c0:c1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bb, :, :], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, i0, i1, j0, j1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("alkd,ijbcld->ijkabc", W_vOoV[a0:a1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :bb, :bc, :, :], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, i0, i1, j0, j1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("blkd,ijacld->ijkabc", W_vOoV[b0:b1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bc, :, :], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp_2, i0, i1, j0, j1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("clkd,ijabld->ijkabc", W_vOoV[c0:c1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bb, :, :], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)

                            _update_packed_aaa_(mycc, r3aaa, r3_tmp, i0, i1, j0, j1, k0, k1,
                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None

    time1 = log.timer_debug1('t3: r3aaa', *time1)
    return r3aaa

def compute_r3bbb_tri_uhf(mycc, imds, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    t2bb = t2[2]
    t3bba, t3bbb = t3[2:4]

    F_OO, F_VV = imds.F_OO, imds.F_VV
    W_OOOO, W_OVOO, W_VVVO, W_VVVV = imds.W_OOOO, imds.W_OVOO, imds.W_VVVO, imds.W_VVVV
    W_VoOv, W_VOOV = imds.W_VoOv, imds.W_VOOV

    r3bbb = np.zeros_like(t3bbb)

    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    t3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=t3bbb.dtype)
    t3_tmp_2 = np.empty((blksize_o_aaa,) * 2 + (blksize_v_aaa,) * 2 + (nocca,) + (nvira,), dtype=t3bbb.dtype)
    for k0, k1 in lib.prange(0, noccb, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvirb, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            bijkabc = (slice(None, bi), slice(None, bj), slice(None, bk),
                                        slice(None, ba), slice(None, bb), slice(None, bc))

                            einsum("bcdk,ijad->ijkabc", W_VVVO[b0:b1, c0:c1, :, k0:k1],
                                t2bb[i0:i1, j0:j1, a0:a1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=0.0)
                            einsum("cbdk,ijad->ijkabc", W_VVVO[c0:c1, b0:b1, :, k0:k1],
                                t2bb[i0:i1, j0:j1, a0:a1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("acdk,ijbd->ijkabc", W_VVVO[a0:a1, c0:c1, :, k0:k1],
                                t2bb[i0:i1, j0:j1, b0:b1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("abdk,ijcd->ijkabc", W_VVVO[a0:a1, b0:b1, :, k0:k1],
                                t2bb[i0:i1, j0:j1, c0:c1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("cadk,ijbd->ijkabc", W_VVVO[c0:c1, a0:a1, :, k0:k1],
                                t2bb[i0:i1, j0:j1, b0:b1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("badk,ijcd->ijkabc", W_VVVO[b0:b1, a0:a1, :, k0:k1],
                                t2bb[i0:i1, j0:j1, c0:c1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("bcdj,ikad->ijkabc", W_VVVO[b0:b1, c0:c1, :, j0:j1],
                                t2bb[i0:i1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("cbdj,ikad->ijkabc", W_VVVO[c0:c1, b0:b1, :, j0:j1],
                                t2bb[i0:i1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("acdj,ikbd->ijkabc", W_VVVO[a0:a1, c0:c1, :, j0:j1],
                                t2bb[i0:i1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("abdj,ikcd->ijkabc", W_VVVO[a0:a1, b0:b1, :, j0:j1],
                                t2bb[i0:i1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("cadj,ikbd->ijkabc", W_VVVO[c0:c1, a0:a1, :, j0:j1],
                                t2bb[i0:i1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("badj,ikcd->ijkabc", W_VVVO[b0:b1, a0:a1, :, j0:j1],
                                t2bb[i0:i1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("bcdi,jkad->ijkabc", W_VVVO[b0:b1, c0:c1, :, i0:i1],
                                t2bb[j0:j1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("cbdi,jkad->ijkabc", W_VVVO[c0:c1, b0:b1, :, i0:i1],
                                t2bb[j0:j1, k0:k1, a0:a1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("acdi,jkbd->ijkabc", W_VVVO[a0:a1, c0:c1, :, i0:i1],
                                t2bb[j0:j1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("abdi,jkcd->ijkabc", W_VVVO[a0:a1, b0:b1, :, i0:i1],
                                t2bb[j0:j1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("cadi,jkbd->ijkabc", W_VVVO[c0:c1, a0:a1, :, i0:i1],
                                t2bb[j0:j1, k0:k1, b0:b1, :], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("badi,jkcd->ijkabc", W_VVVO[b0:b1, a0:a1, :, i0:i1],
                                t2bb[j0:j1, k0:k1, c0:c1, :], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)

                            einsum("lcjk,ilab->ijkabc", W_OVOO[:, c0:c1, j0:j1, k0:k1],
                                t2bb[i0:i1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lbjk,ilac->ijkabc", W_OVOO[:, b0:b1, j0:j1, k0:k1],
                                t2bb[i0:i1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lajk,ilbc->ijkabc", W_OVOO[:, a0:a1, j0:j1, k0:k1],
                                t2bb[i0:i1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lckj,ilab->ijkabc", W_OVOO[:, c0:c1, k0:k1, j0:j1],
                                t2bb[i0:i1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lbkj,ilac->ijkabc", W_OVOO[:, b0:b1, k0:k1, j0:j1],
                                t2bb[i0:i1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lakj,ilbc->ijkabc", W_OVOO[:, a0:a1, k0:k1, j0:j1],
                                t2bb[i0:i1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lcik,jlab->ijkabc", W_OVOO[:, c0:c1, i0:i1, k0:k1],
                                t2bb[j0:j1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lbik,jlac->ijkabc", W_OVOO[:, b0:b1, i0:i1, k0:k1],
                                t2bb[j0:j1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("laik,jlbc->ijkabc", W_OVOO[:, a0:a1, i0:i1, k0:k1],
                                t2bb[j0:j1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lcij,klab->ijkabc", W_OVOO[:, c0:c1, i0:i1, j0:j1],
                                t2bb[k0:k1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lbij,klac->ijkabc", W_OVOO[:, b0:b1, i0:i1, j0:j1],
                                t2bb[k0:k1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("laij,klbc->ijkabc", W_OVOO[:, a0:a1, i0:i1, j0:j1],
                                t2bb[k0:k1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lcki,jlab->ijkabc", W_OVOO[:, c0:c1, k0:k1, i0:i1],
                                t2bb[j0:j1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lbki,jlac->ijkabc", W_OVOO[:, b0:b1, k0:k1, i0:i1],
                                t2bb[j0:j1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("laki,jlbc->ijkabc", W_OVOO[:, a0:a1, k0:k1, i0:i1],
                                t2bb[j0:j1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("lcji,klab->ijkabc", W_OVOO[:, c0:c1, j0:j1, i0:i1],
                                t2bb[k0:k1, :, a0:a1, b0:b1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                            einsum("lbji,klac->ijkabc", W_OVOO[:, b0:b1, j0:j1, i0:i1],
                                t2bb[k0:k1, :, a0:a1, c0:c1], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                            einsum("laji,klbc->ijkabc", W_OVOO[:, a0:a1, j0:j1, i0:i1],
                                t2bb[k0:k1, :, b0:b1, c0:c1], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)

                            for d0, d1 in lib.prange(0, nvirb, blksize_v_aaa):
                                bd = d1 - d0
                                _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, d0, d1)
                                einsum("cd,ijkabd->ijkabc", F_VV[c0:c1, d0:d1],
                                    t3_tmp[:bi, :bj, :bk, :ba, :bb, :bd], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, c0, c1, d0, d1)
                                einsum("bd,ijkacd->ijkabc", F_VV[b0:b1, d0:d1],
                                    t3_tmp[:bi, :bj, :bk, :ba, :bc, :bd], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, b0, b1, c0, c1, d0, d1)
                                einsum("ad,ijkbcd->ijkabc", F_VV[a0:a1, d0:d1],
                                    t3_tmp[:bi, :bj, :bk, :bb, :bc, :bd], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                for e0, e1 in lib.prange(0, nvirb, blksize_v_aaa):
                                    be = e1 - e0
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, c0, c1)
                                    einsum("abde,ijkdec->ijkabc", W_VVVV[a0:a1, b0:b1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bc], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, b0, b1)
                                    einsum("acde,ijkdeb->ijkabc", W_VVVV[a0:a1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :bb], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, k0, k1, d0, d1, e0, e1, a0, a1)
                                    einsum("bcde,ijkdea->ijkabc", W_VVVV[b0:b1, c0:c1, d0:d1, e0:e1],
                                        t3_tmp[:bi, :bj, :bk, :bd, :be, :ba], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)

                            for l0, l1 in lib.prange(0, noccb, blksize_o_aaa):
                                bl = l1 - l0
                                _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, j0, j1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum("lk,ijlabc->ijkabc", F_OO[l0:l1, k0:k1],
                                    t3_tmp[:bi, :bj, :bl, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                _unp_bbb_(mycc, t3bbb, t3_tmp, i0, i1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum("lj,iklabc->ijkabc", F_OO[l0:l1, j0:j1],
                                    t3_tmp[:bi, :bk, :bl, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                _unp_bbb_(mycc, t3bbb, t3_tmp, j0, j1, k0, k1, l0, l1, a0, a1, b0, b1, c0, c1)
                                einsum("li,jklabc->ijkabc", F_OO[l0:l1, i0:i1],
                                    t3_tmp[:bj, :bk, :bl, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                for m0, m1 in lib.prange(0, noccb, blksize_o_aaa):
                                    bm = m1 - m0
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, k0, k1, a0, a1, b0, b1, c0, c1)
                                    einsum("lmij,lmkabc->ijkabc", W_OOOO[l0:l1, m0:m1, i0:i1, j0:j1],
                                        t3_tmp[:bl, :bm, :bk, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, j0, j1, a0, a1, b0, b1, c0, c1)
                                    einsum("lmik,lmjabc->ijkabc", W_OOOO[l0:l1, m0:m1, i0:i1, k0:k1],
                                        t3_tmp[:bl, :bm, :bj, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=-0.5, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, m0, m1, i0, i1, a0, a1, b0, b1, c0, c1)
                                    einsum("lmjk,lmiabc->ijkabc", W_OOOO[l0:l1, m0:m1, j0:j1, k0:k1],
                                        t3_tmp[:bl, :bm, :bi, :ba, :bb, :bc], out=r3_tmp[bijkabc], alpha=0.5, beta=1.0)

                            for l0, l1 in lib.prange(0, noccb, blksize_o_aaa):
                                bl = l1 - l0
                                for d0, d1 in lib.prange(0, nvirb, blksize_v_aaa):
                                    bd = d1 - d0
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum("alid,ljkdbc->ijkabc", W_VOOV[a0:a1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum("blid,ljkdac->ijkabc", W_VOOV[b0:b1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, j0, j1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum("clid,ljkdab->ijkabc", W_VOOV[c0:c1, l0:l1, i0:i1, d0:d1],
                                        t3_tmp[:bl, :bj, :bk, :bd, :ba, :bb], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, b0, b1, c0, c1)
                                    einsum("aljd,likdbc->ijkabc", W_VOOV[a0:a1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :bb, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, c0, c1)
                                    einsum("bljd,likdac->ijkabc", W_VOOV[b0:b1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, k0, k1, d0, d1, a0, a1, b0, b1)
                                    einsum("cljd,likdab->ijkabc", W_VOOV[c0:c1, l0:l1, j0:j1, d0:d1],
                                        t3_tmp[:bl, :bi, :bk, :bd, :ba, :bb], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, b0, b1, c0, c1)
                                    einsum("alkd,lijdbc->ijkabc", W_VOOV[a0:a1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, c0, c1)
                                    einsum("blkd,lijdac->ijkabc", W_VOOV[b0:b1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                                    _unp_bbb_(mycc, t3bbb, t3_tmp, l0, l1, i0, i1, j0, j1, d0, d1, a0, a1, b0, b1)
                                    einsum("clkd,lijdab->ijkabc", W_VOOV[c0:c1, l0:l1, k0:k1, d0:d1],
                                        t3_tmp[:bl, :bi, :bj, :bd, :ba, :bb], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)

                            _unp_bba_(mycc, t3bba, t3_tmp_2, j0, j1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("alid,jkbcld->ijkabc", W_VoOv[a0:a1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, j0, j1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("blid,jkacld->ijkabc", W_VoOv[b0:b1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, j0, j1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("clid,jkabld->ijkabc", W_VoOv[c0:c1, :, i0:i1, :],
                                t3_tmp_2[:bj, :bk, :ba, :bb], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, i0, i1, k0, k1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("aljd,ikbcld->ijkabc", W_VoOv[a0:a1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :bb, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, i0, i1, k0, k1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("bljd,ikacld->ijkabc", W_VoOv[b0:b1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, i0, i1, k0, k1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("cljd,ikabld->ijkabc", W_VoOv[c0:c1, :, j0:j1, :],
                                t3_tmp_2[:bi, :bk, :ba, :bb], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, i0, i1, j0, j1, b0, b1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("alkd,ijbcld->ijkabc", W_VoOv[a0:a1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :bb, :bc], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, i0, i1, j0, j1, a0, a1, c0, c1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("blkd,ijacld->ijkabc", W_VoOv[b0:b1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bc], out=r3_tmp[bijkabc], alpha=-1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp_2, i0, i1, j0, j1, a0, a1, b0, b1,
                                blk_i=blksize_o_aaa, blk_j=blksize_o_aaa, blk_a=blksize_v_aaa, blk_b=blksize_v_aaa)
                            einsum("clkd,ijabld->ijkabc", W_VoOv[c0:c1, :, k0:k1, :],
                                t3_tmp_2[:bi, :bj, :ba, :bb], out=r3_tmp[bijkabc], alpha=1.0, beta=1.0)

                            _update_packed_bbb_(mycc, r3bbb, r3_tmp, i0, i1, j0, j1, k0, k1,
                                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None

    time1 = log.timer_debug1('t3: r3bbb', *time1)
    return r3bbb

def compute_r3aab_tri_uhf(mycc, imds, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    t2aa, t2ab = t2[0:2]
    t3aaa, t3aab, t3bba = t3[0:3]

    F_oo, F_vv, F_OO, F_VV = imds.F_oo, imds.F_vv, imds.F_OO, imds.F_VV
    W_oooo, W_oOoO, W_ovoo, W_oVoO = imds.W_oooo, imds.W_oOoO, imds.W_ovoo, imds.W_oVoO
    W_vOoO, W_oVoV, W_vOvO, W_vVoV = imds.W_vOoO, imds.W_oVoV, imds.W_vOvO, imds.W_vVoV
    W_voov, W_vOoV, W_VoOv, W_VOOV = imds.W_voov, imds.W_vOoV, imds.W_VoOv, imds.W_VOOV
    W_vvvo, W_vVvO, W_vvvv, W_vVvV = imds.W_vvvo, imds.W_vVvO, imds.W_vvvv, imds.W_vVvV

    r3aab = np.zeros_like(t3aab)

    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aaa.dtype)
    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=t3aaa.dtype)
    t3_tmp_2 = np.empty((blksize_o_aab,) + (noccb,) + (blksize_v_aab,) + (nvirb,)
                        + (nocca,) + (nvira,), dtype=t3aaa.dtype)
    t3_tmp_3 = np.empty((blksize_o_aab,) * 2 + (nocca,) + (blksize_v_aab,) * 2 + (nvira,), dtype=t3aaa.dtype)
    for j0, j1 in lib.prange(0, nocca, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvira, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0

                    einsum("bcdk,ijad->ijabkc", W_vVvO[b0:b1], t2aa[i0:i1, j0:j1, a0:a1, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=0.0)
                    einsum("acdk,ijbd->ijabkc", W_vVvO[a0:a1], t2aa[i0:i1, j0:j1, b0:b1, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    einsum("lcjk,ilab->ijabkc", W_oVoO[:, :, j0:j1, :], t2aa[i0:i1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("lcik,jlab->ijabkc", W_oVoO[:, :, i0:i1, :], t2aa[j0:j1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    einsum("bcjd,iakd->ijabkc", W_vVoV[b0:b1, :, j0:j1, :], t2ab[i0:i1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum("acjd,ibkd->ijabkc", W_vVoV[a0:a1, :, j0:j1, :], t2ab[i0:i1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("bcid,jakd->ijabkc", W_vVoV[b0:b1, :, i0:i1, :], t2ab[j0:j1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("acid,jbkd->ijabkc", W_vVoV[a0:a1, :, i0:i1, :], t2ab[j0:j1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    einsum("abdi,jdkc->ijabkc", W_vvvo[a0:a1, b0:b1, :, i0:i1], t2ab[j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum("badi,jdkc->ijabkc", W_vvvo[b0:b1, a0:a1, :, i0:i1], t2ab[j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum("abdj,idkc->ijabkc", W_vvvo[a0:a1, b0:b1, :, j0:j1], t2ab[i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum("badj,idkc->ijabkc", W_vvvo[b0:b1, a0:a1, :, j0:j1], t2ab[i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)

                    einsum("aljk,iblc->ijabkc", W_vOoO[a0:a1, :, j0:j1, :], t2ab[i0:i1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum("bljk,ialc->ijabkc", W_vOoO[b0:b1, :, j0:j1, :], t2ab[i0:i1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("alik,jblc->ijabkc", W_vOoO[a0:a1, :, i0:i1, :], t2ab[j0:j1, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("blik,jalc->ijabkc", W_vOoO[b0:b1, :, i0:i1, :], t2ab[j0:j1, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    einsum("laij,lbkc->ijabkc", W_ovoo[:, a0:a1, i0:i1, j0:j1], t2ab[:, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum("lbij,lakc->ijabkc", W_ovoo[:, b0:b1, i0:i1, j0:j1], t2ab[:, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum("laji,lbkc->ijabkc", W_ovoo[:, a0:a1, j0:j1, i0:i1], t2ab[:, b0:b1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum("lbji,lakc->ijabkc", W_ovoo[:, b0:b1, j0:j1, i0:i1], t2ab[:, a0:a1, :, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    _unp_aab_(mycc, t3aab, t3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    einsum("cd,ijabkd->ijabkc", F_VV, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum("lk,ijablc->ijabkc", F_OO, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("clkd,ijabld->ijabkc", W_VOOV, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                        bd = d1 - d0
                        _unp_aab_(mycc, t3aab, t3_tmp, i0, i1, j0, j1, b0, b1, d0, d1)
                        einsum("ad,ijbdkc->ijabkc", F_vv[a0:a1, d0:d1],
                            t3_tmp[:bi, :bj, :bb, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("acde,ijbdke->ijabkc", W_vVvV[a0:a1, :, d0:d1, :],
                            t3_tmp[:bi, :bj, :bb, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("aldk,ijbdlc->ijabkc", W_vOvO[a0:a1, :, d0:d1, :],
                            t3_tmp[:bi, :bj, :bb, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        _unp_aab_(mycc, t3aab, t3_tmp, i0, i1, j0, j1, a0, a1, d0, d1)
                        einsum("bd,ijadkc->ijabkc", F_vv[b0:b1, d0:d1],
                            t3_tmp[:bi, :bj, :ba, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("bcde,ijadke->ijabkc", W_vVvV[b0:b1, :, d0:d1, :],
                            t3_tmp[:bi, :bj, :ba, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("bldk,ijadlc->ijabkc", W_vOvO[b0:b1, :, d0:d1, :],
                            t3_tmp[:bi, :bj, :ba, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        _unp_aab_(mycc, t3aab, t3_tmp, j0, j1, l0, l1, a0, a1, b0, b1)
                        einsum("li,jlabkc->ijabkc", F_oo[l0:l1, i0:i1],
                            t3_tmp[:bj, :bl, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("lmik,jlabmc->ijabkc", W_oOoO[l0:l1, :, i0:i1, :],
                            t3_tmp[:bj, :bl, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("lcid,jlabkd->ijabkc", W_oVoV[l0:l1, :, i0:i1, :],
                            t3_tmp[:bj, :bl, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        _unp_aab_(mycc, t3aab, t3_tmp, i0, i1, l0, l1, a0, a1, b0, b1)
                        einsum("lj,ilabkc->ijabkc", F_oo[l0:l1, j0:j1],
                            t3_tmp[:bi, :bl, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("lmjk,ilabmc->ijabkc", W_oOoO[l0:l1, :, j0:j1, :],
                            t3_tmp[:bi, :bl, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("lcjd,ilabkd->ijabkc", W_oVoV[l0:l1, :, j0:j1, :],
                            t3_tmp[:bi, :bl, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                        bd = d1 - d0
                        for e0, e1 in lib.prange(0, nvira, blksize_v_aab):
                            be = e1 - e0
                            _unp_aab_(mycc, t3aab, t3_tmp, i0, i1, j0, j1, d0, d1, e0, e1)
                            einsum("abde,ijdekc->ijabkc", W_vvvv[a0:a1, b0:b1, d0:d1, e0:e1],
                                t3_tmp[:bi, :bj, :bd, :be], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        for m0, m1 in lib.prange(0, nocca, blksize_o_aab):
                            bm = m1 - m0
                            _unp_aab_(mycc, t3aab, t3_tmp, l0, l1, m0, m1, a0, a1, b0, b1)
                            einsum("lmij,lmabkc->ijabkc", W_oooo[l0:l1, m0:m1, i0:i1, j0:j1],
                                t3_tmp[:bl, :bm, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                            bd = d1 - d0
                            _unp_aab_(mycc, t3aab, t3_tmp, l0, l1, j0, j1, d0, d1, b0, b1)
                            einsum("alid,ljdbkc->ijabkc", W_voov[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp, l0, l1, j0, j1, d0, d1, a0, a1)
                            einsum("blid,ljdakc->ijabkc", W_voov[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp, l0, l1, i0, i1, d0, d1, b0, b1)
                            einsum("aljd,lidbkc->ijabkc", W_voov[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_aab_(mycc, t3aab, t3_tmp, l0, l1, i0, i1, d0, d1, a0, a1)
                            einsum("bljd,lidakc->ijabkc", W_voov[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    # TODO: This unpacking performs redundant work; optimize to avoid repeated operations
                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                            bd = d1 - d0
                            _unp_bba_(mycc, t3bba, t3_tmp_2, l0, l1, 0, noccb,
                                    d0, d1, 0, nvirb, blk_j=noccb, blk_b=nvirb)
                            einsum("alid,lkdcjb->ijabkc", W_vOoV[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            einsum("blid,lkdcja->ijabkc", W_vOoV[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum("aljd,lkdcib->ijabkc", W_vOoV[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum("bljd,lkdcia->ijabkc", W_vOoV[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    _unp_aaa_(mycc, t3aaa, t3_tmp_3, i0, i1, j0, j1, 0, nocca, a0, a1, b0, b1, 0, nvira,
                            blk_i=blksize_o_aab, blk_j=blksize_o_aab, blk_k=nocca,
                            blk_a=blksize_v_aab, blk_b=blksize_v_aab, blk_c=nvira)
                    einsum("clkd,ijlabd->ijabkc", W_VoOv, t3_tmp_3[:bi, :bj, :, :ba, :bb, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    _update_packed_aab_(mycc, r3aab, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None
    t3_tmp_3 = None
    W_vvvo = imds.W_vvvo = None
    W_ovoo = imds.W_ovoo = None
    W_vvvv = imds.W_vvvv = None
    W_oooo = imds.W_oooo = None

    time1 = log.timer_debug1('t3: r3aab', *time1)
    return r3aab

def compute_r3bba_tri_uhf(mycc, imds, t2, t3):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    t2ab, t2bb = t2[1:3]
    t3aab, t3bba, t3bbb = t3[1:4]

    F_oo, F_vv, F_OO, F_VV = imds.F_oo, imds.F_vv, imds.F_OO, imds.F_VV
    W_oOoO, W_OOOO, W_oVoO, W_OVOO = imds.W_oOoO, imds.W_OOOO, imds.W_oVoO, imds.W_OVOO
    W_vOoO, W_oVoV, W_vOvO, W_vVoV = imds.W_vOoO, imds.W_oVoV, imds.W_vOvO, imds.W_vVoV
    W_voov, W_vOoV, W_VoOv, W_VOOV = imds.W_voov, imds.W_vOoV, imds.W_VoOv, imds.W_VOOV
    W_vVvO, W_VVVO, W_vVvV, W_VVVV = imds.W_vVvO, imds.W_VVVO, imds.W_vVvV, imds.W_VVVV

    r3bba = np.zeros_like(t3bba)

    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bbb.dtype)
    t3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=t3bbb.dtype)
    t3_tmp_2 = np.empty((blksize_o_aab,) + (nocca,) + (blksize_v_aab,)
                        + (nvira,) + (noccb,) + (nvirb,), dtype=t3bbb.dtype)
    t3_tmp_3 = np.empty((blksize_o_aab,) * 2 + (noccb,) + (blksize_v_aab,) * 2 + (nvirb,), dtype=t3bbb.dtype)
    for j0, j1 in lib.prange(0, noccb, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvirb, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0

                    einsum("cbkd,ijad->ijabkc", W_vVoV[:, b0:b1], t2bb[i0:i1, j0:j1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=0.0)
                    einsum("cakd,ijbd->ijabkc", W_vVoV[:, a0:a1], t2bb[i0:i1, j0:j1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    einsum("clkj,ilab->ijabkc", W_vOoO[..., j0:j1], t2bb[i0:i1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("clki,jlab->ijabkc", W_vOoO[..., i0:i1], t2bb[j0:j1, :, a0:a1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    einsum("cbdj,kdia->ijabkc", W_vVvO[:, b0:b1, :, j0:j1], t2ab[:, :, i0:i1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum("cadj,kdib->ijabkc", W_vVvO[:, a0:a1, :, j0:j1], t2ab[:, :, i0:i1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("cbdi,kdja->ijabkc", W_vVvO[:, b0:b1, :, i0:i1], t2ab[:, :, j0:j1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("cadi,kdjb->ijabkc", W_vVvO[:, a0:a1, :, i0:i1], t2ab[:, :, j0:j1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    einsum("abdi,kcjd->ijabkc", W_VVVO[a0:a1, b0:b1, :, i0:i1], t2ab[:, :, j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum("badi,kcjd->ijabkc", W_VVVO[b0:b1, a0:a1, :, i0:i1], t2ab[:, :, j0:j1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum("abdj,kcid->ijabkc", W_VVVO[a0:a1, b0:b1, :, j0:j1], t2ab[:, :, i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum("badj,kcid->ijabkc", W_VVVO[b0:b1, a0:a1, :, j0:j1], t2ab[:, :, i0:i1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)

                    einsum("lakj,lcib->ijabkc", W_oVoO[:, a0:a1, :, j0:j1], t2ab[:, :, i0:i1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum("lbkj,lcia->ijabkc", W_oVoO[:, b0:b1, :, j0:j1], t2ab[:, :, i0:i1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("laki,lcjb->ijabkc", W_oVoO[:, a0:a1, :, i0:i1], t2ab[:, :, j0:j1, b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("lbki,lcja->ijabkc", W_oVoO[:, b0:b1, :, i0:i1], t2ab[:, :, j0:j1, a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    einsum("laij,kclb->ijabkc", W_OVOO[:, a0:a1, i0:i1, j0:j1], t2ab[..., b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)
                    einsum("lbij,kcla->ijabkc", W_OVOO[:, b0:b1, i0:i1, j0:j1], t2ab[..., a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum("laji,kclb->ijabkc", W_OVOO[:, a0:a1, j0:j1, i0:i1], t2ab[..., b0:b1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-0.5, beta=1.0)
                    einsum("lbji,kcla->ijabkc", W_OVOO[:, b0:b1, j0:j1, i0:i1], t2ab[..., a0:a1],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    _unp_bba_(mycc, t3bba, t3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    einsum("cd,ijabkd->ijabkc", F_vv, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                    einsum("lk,ijablc->ijabkc", F_oo, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                    einsum("clkd,ijabld->ijabkc", W_voov, t3_tmp[:bi, :bj, :ba, :bb],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                        bd = d1 - d0
                        _unp_bba_(mycc, t3bba, t3_tmp, i0, i1, j0, j1, b0, b1, d0, d1)
                        einsum("ad,ijbdkc->ijabkc", F_VV[a0:a1, d0:d1],
                            t3_tmp[:bi, :bj, :bb, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("caed,ijbdke->ijabkc", W_vVvV[:, a0:a1, :, d0:d1],
                            t3_tmp[:bi, :bj, :bb, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("lakd,ijbdlc->ijabkc", W_oVoV[:, a0:a1, :, d0:d1],
                            t3_tmp[:bi, :bj, :bb, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        _unp_bba_(mycc,t3bba, t3_tmp, i0, i1, j0, j1, a0, a1, d0, d1)
                        einsum("bd,ijadkc->ijabkc", F_VV[b0:b1, d0:d1],
                            t3_tmp[:bi, :bj, :ba, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("cbed,ijadke->ijabkc", W_vVvV[:, b0:b1, :, d0:d1],
                            t3_tmp[:bi, :bj, :ba, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("lbkd,ijadlc->ijabkc", W_oVoV[:, b0:b1, :, d0:d1],
                            t3_tmp[:bi, :bj, :ba, :bd], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)

                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, j0, j1, a0, a1, b0, b1)
                        einsum("li,ljabkc->ijabkc", F_OO[l0:l1, i0:i1],
                            t3_tmp[:bl, :bj, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("mlki,ljabmc->ijabkc", W_oOoO[:, l0:l1, :, i0:i1],
                            t3_tmp[:bl, :bj, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("cldi,ljabkd->ijabkc", W_vOvO[:, l0:l1, :, i0:i1],
                            t3_tmp[:bl, :bj, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, i0, i1, a0, a1, b0, b1)
                        einsum("lj,liabkc->ijabkc", F_OO[l0:l1, j0:j1],
                            t3_tmp[:bl, :bi, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                        einsum("mlkj,liabmc->ijabkc", W_oOoO[:, l0:l1, :, j0:j1],
                            t3_tmp[:bl, :bi, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                        einsum("cldj,liabkd->ijabkc", W_vOvO[:, l0:l1, :, j0:j1],
                            t3_tmp[:bl, :bi, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb, :, :], alpha=1.0, beta=1.0)

                    for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                        bd = d1 - d0
                        for e0, e1 in lib.prange(0, nvirb, blksize_v_aab):
                            be = e1 - e0
                            _unp_bba_(mycc, t3bba, t3_tmp, i0, i1, j0, j1, d0, d1, e0, e1)
                            einsum("abde,ijdekc->ijabkc", W_VVVV[a0:a1, b0:b1, d0:d1, e0:e1],
                                t3_tmp[:bi, :bj, :bd, :be], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        for m0, m1 in lib.prange(0, noccb, blksize_o_aab):
                            bm = m1 - m0
                            _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, m0, m1, a0, a1, b0, b1)
                            einsum("lmij,lmabkc->ijabkc", W_OOOO[l0:l1, m0:m1, i0:i1, j0:j1],
                                t3_tmp[:bl, :bm, :ba, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=0.5, beta=1.0)

                    for l0, l1 in lib.prange(0, noccb, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvirb, blksize_v_aab):
                            bd = d1 - d0
                            _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, j0, j1, d0, d1, b0, b1)
                            einsum("alid,ljdbkc->ijabkc", W_VOOV[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, j0, j1, d0, d1, a0, a1)
                            einsum("blid,ljdakc->ijabkc", W_VOOV[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp[:bl, :bj, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, i0, i1, d0, d1, b0, b1)
                            einsum("aljd,lidbkc->ijabkc", W_VOOV[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :bb], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            _unp_bba_(mycc, t3bba, t3_tmp, l0, l1, i0, i1, d0, d1, a0, a1)
                            einsum("bljd,lidakc->ijabkc", W_VOOV[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp[:bl, :bi, :bd, :ba], out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    # TODO: This unpacking performs redundant work; optimize to avoid repeated operations
                    for l0, l1 in lib.prange(0, nocca, blksize_o_aab):
                        bl = l1 - l0
                        for d0, d1 in lib.prange(0, nvira, blksize_v_aab):
                            bd = d1 - d0
                            _unp_aab_(mycc, t3aab, t3_tmp_2, l0, l1, 0, nocca,
                                    d0, d1, 0, nvira, blk_j=nocca, blk_b=nvira)
                            einsum("alid,lkdcjb->ijabkc", W_VoOv[a0:a1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)
                            einsum("blid,lkdcja->ijabkc", W_VoOv[b0:b1, l0:l1, i0:i1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, j0:j1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum("aljd,lkdcib->ijabkc", W_VoOv[a0:a1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, b0:b1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=-1.0, beta=1.0)
                            einsum("bljd,lkdcia->ijabkc", W_VoOv[b0:b1, l0:l1, j0:j1, d0:d1],
                                t3_tmp_2[:bl, :, :bd, :, i0:i1, a0:a1],
                                out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    _unp_bbb_(mycc, t3bbb, t3_tmp_3, i0, i1, j0, j1, 0, noccb, a0, a1, b0, b1, 0, nvirb,
                        blk_i=blksize_o_aab, blk_j=blksize_o_aab, blk_k=noccb,
                        blk_a=blksize_v_aab, blk_b=blksize_v_aab, blk_c=nvirb)
                    einsum("clkd,ijlabd->ijabkc", W_vOoV, t3_tmp_3[:bi, :bj, :, :ba, :bb, :],
                        out=r3_tmp[:bi, :bj, :ba, :bb], alpha=1.0, beta=1.0)

                    _update_packed_bba_(mycc, r3bba, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    t3_tmp = None
    t3_tmp_2 = None
    t3_tmp_3 = None
    F_oo = imds.F_oo = None
    F_OO = imds.F_OO = None
    F_vv = imds.F_vv = None
    F_VV = imds.F_VV = None
    W_vVoV = imds.W_vVoV = None
    W_vVvO = imds.W_vVvO = None
    W_voov = imds.W_voov = None
    W_vOoV = imds.W_vOoV = None
    W_oVoO = imds.W_oVoO = None
    W_vOoO = imds.W_vOoO = None
    W_vVvV = imds.W_vVvV = None
    W_oOoO = imds.W_oOoO = None
    W_oVoV = imds.W_oVoV = None
    W_vOvO = imds.W_vOvO = None
    W_VoOv = imds.W_VoOv = None
    W_VOOV = imds.W_VOOV = None
    W_VVVV = imds.W_VVVV = None
    W_VVVO = imds.W_VVVO = None
    W_OVOO = imds.W_OVOO = None
    W_OOOO = imds.W_OOOO = None

    time1 = log.timer_debug1('t3: r3bba', *time1)
    return r3bba

def compute_r3_tri_uhf(mycc, imds, t2, t3):
    '''Compute r3 with triangular-stored T3 amplitudes; r3 is returned in triangular form as well.'''
    r3aaa = compute_r3aaa_tri_uhf(mycc, imds, t2, t3)
    r3bbb = compute_r3bbb_tri_uhf(mycc, imds, t2, t3)
    r3aab = compute_r3aab_tri_uhf(mycc, imds, t2, t3)
    r3bba = compute_r3bba_tri_uhf(mycc, imds, t2, t3)
    r3 = [r3aaa, r3aab, r3bba, r3bbb]
    return r3

def r3_tri_divide_e_uhf_(mycc, r3, mo_energy):
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    blksize_o_aaa, blksize_v_aaa = mycc.blksize_o_aaa, mycc.blksize_v_aaa
    blksize_o_aab, blksize_v_aab = mycc.blksize_o_aab, mycc.blksize_v_aab
    eia_a = mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:] - mycc.level_shift
    eia_b = mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:] - mycc.level_shift

    r3aaa, r3aab, r3bba, r3bbb = r3

    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    for k0, k1 in lib.prange(0, nocca, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvira, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            _unp_aaa_(mycc, r3aaa, r3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1)
                            eijkabc_blk = (eia_a[i0:i1, None, None, a0:a1, None, None]
                                        + eia_a[None, j0:j1, None, None, b0:b1, None]
                                        + eia_a[None, None, k0:k1, None, None, c0:c1])
                            r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc] /= eijkabc_blk
                            _update_packed_aaa_(mycc, r3aaa, r3_tmp, i0, i1, j0, j1, k0, k1,
                                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    eijkabc_blk = None

    r3_tmp = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aaa,) * 3 + (blksize_v_aaa,) * 3, dtype=r3aaa.dtype)
    for k0, k1 in lib.prange(0, noccb, blksize_o_aaa):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1 - 1, blksize_o_aaa):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aaa):
                bi = i1 - i0
                for c0, c1 in lib.prange(0, nvirb, blksize_v_aaa):
                    bc = c1 - c0
                    for b0, b1 in lib.prange(0, c1 - 1, blksize_v_aaa):
                        bb = b1 - b0
                        for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aaa):
                            ba = a1 - a0
                            _unp_bbb_(mycc, r3bbb, r3_tmp, i0, i1, j0, j1, k0, k1, a0, a1, b0, b1, c0, c1)
                            eijkabc_blk = (eia_b[i0:i1, None, None, a0:a1, None, None]
                                        + eia_b[None, j0:j1, None, None, b0:b1, None]
                                        + eia_b[None, None, k0:k1, None, None, c0:c1])
                            r3_tmp[:bi, :bj, :bk, :ba, :bb, :bc] /= eijkabc_blk
                            _update_packed_bbb_(mycc, r3bbb, r3_tmp, i0, i1, j0, j1, k0, k1,
                                                a0, a1, b0, b1, c0, c1, alpha=1.0, beta=0.0)
    r3_tmp = None
    eijkabc_blk = None

    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (noccb,) + (nvirb,), dtype=r3aaa.dtype)
    for j0, j1 in lib.prange(0, nocca, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvira, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0
                    _unp_aab_(mycc, r3aab, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    eijkabc_blk = (eia_a[i0:i1, None, a0:a1, None, None, None]
                            + eia_a[None, j0:j1, None, b0:b1, None, None] + eia_b[None, None, None, None, :, :])
                    r3_tmp[:bi, :bj, :ba, :bb] /= eijkabc_blk
                    _update_packed_aab_(mycc, r3aab, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    eijkabc_blk = None

    r3_tmp = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=r3aaa.dtype)
    eijkabc_blk = np.empty((blksize_o_aab,) * 2 + (blksize_v_aab,) * 2 + (nocca,) + (nvira,), dtype=r3aaa.dtype)
    for j0, j1 in lib.prange(0, noccb, blksize_o_aab):
        bj = j1 - j0
        for i0, i1 in lib.prange(0, j1 - 1, blksize_o_aab):
            bi = i1 - i0
            for b0, b1 in lib.prange(0, nvirb, blksize_v_aab):
                bb = b1 - b0
                for a0, a1 in lib.prange(0, b1 - 1, blksize_v_aab):
                    ba = a1 - a0
                    _unp_bba_(mycc, r3bba, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
                    eijkabc_blk = (eia_b[i0:i1, None, a0:a1, None, None, None]
                            + eia_b[None, j0:j1, None, b0:b1, None, None] + eia_a[None, None, None, None, :, :])
                    r3_tmp[:bi, :bj, :ba, :bb] /= eijkabc_blk
                    _update_packed_bba_(mycc, r3bba, r3_tmp, i0, i1, j0, j1, a0, a1, b0, b1)
    r3_tmp = None
    eijkabc_blk = None
    return r3

def update_amps_uccsdt_tri_(mycc, tamps, eris):
    '''Update UCCSDT amplitudes in place, with T3 amplitudes stored in triangular form.'''
    assert (isinstance(eris, _PhysicistsERIs))
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1, t2, t3 = tamps
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t3aaa, t3aab, t3bba, t3bbb = t3
    mo_energy = eris.mo_energy

    imds = _IMDS()

    # t1 t2
    update_t1_fock_eris_uhf(mycc, imds, t1, eris)
    time1 = log.timer_debug1('t1t2: update fock and eris', *time0)
    intermediates_t1t2_uhf(mycc, imds, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2_uhf(mycc, imds, t2)
    r1r2_add_t3_tri_uhf_(mycc, imds, r1, r2, t3)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # antisymmetrization
    antisymmetrize_r2_uhf_(r2)
    time1 = log.timer_debug1('t1t2: antisymmetrize r2', *time1)
    # divide by eijkabc
    r1r2_divide_e_uhf_(mycc, r1, r2, mo_energy)
    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2
    time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)

    res_norm = [np.linalg.norm(r1a), np.linalg.norm(r1b),
                np.linalg.norm(r2aa), np.linalg.norm(r2ab), np.linalg.norm(r2bb)]

    t1a += r1a
    t1b += r1b
    t2aa += r2aa
    t2ab += r2ab
    t2bb += r2bb
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)

    # t3
    intermediates_t3_uhf(mycc, imds, t2)
    intermediates_t3_add_t3_tri_uhf(mycc, imds, t3)
    imds.t1_fock, imds.t1_eris = None, None
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3_tri_uhf(mycc, imds, t2, t3)
    imds = None
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # divide by eijkabc
    r3_tri_divide_e_uhf_(mycc, r3, mo_energy)
    r3aaa, r3aab, r3bba, r3bbb = r3
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm += [np.linalg.norm(r3aaa), np.linalg.norm(r3aab), np.linalg.norm(r3bba), np.linalg.norm(r3bbb)]

    t3aaa += r3aaa
    r3aaa = None
    t3bbb += r3bbb
    r3bbb = None
    t3aab += r3aab
    r3aab = None
    t3bba += r3bba
    r3bba = None
    t3 = [t3aaa, t3aab, t3bba, t3bbb]
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)

    tamps = [t1, t2, t3]
    return res_norm

def amplitudes_to_vector_uhf(mycc, tamps):
    '''Convert T-amplitudes to a vector form, storing only symmetry-unique elements (triangular components).'''
    from math import prod, factorial
    nx = lambda nocc, order: prod(nocc - i for i in range(order)) // factorial(order)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    tamps_size = [0]
    for i in range(len(tamps)):
        for na in range(i + 1, -1, -1):
            nb = i + 1 - na
            if na >= nb:
                tamps_size.append(nx(nocca, na) * nx(nvira, na) * nx(noccb, nb) * nx(nvirb, nb))
            else:
                tamps_size.append(nx(noccb, nb) * nx(nvirb, nb) * nx(nocca, na) * nx(nvira, na))
    cum_sizes = np.cumsum(tamps_size)
    vector = np.zeros(cum_sizes[-1], dtype=tamps[0][0].dtype)
    st = 0
    for i, t in enumerate(tamps):
        for j in range(i + 2):
            idx = mycc.unique_tamps_map[i][j]
            vector[cum_sizes[st] : cum_sizes[st + 1]] = t[j][idx].ravel()
            st += 1
    return vector

def vector_to_amplitudes_uhf(mycc, vector):
    '''Reconstruct T-amplitudes from a vector, expanding the stored unique elements into the full tensor.'''
    if mycc.unique_tamps_map is None:
        mycc.unique_tamps_map = mycc.build_unique_tamps_map()

    from math import prod, factorial
    nx = lambda nocc, order: prod(nocc - i for i in range(order)) // factorial(order)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    tamps_size = [0]
    for i in range(mycc.cc_order):
        for na in range(i + 1, -1, -1):
            nb = i + 1 - na
            if na >= nb:
                tamps_size.append(nx(nocca, na) * nx(nvira, na) * nx(noccb, nb) * nx(nvirb, nb))
            else:
                tamps_size.append(nx(noccb, nb) * nx(nvirb, nb) * nx(nocca, na) * nx(nvira, na))
    cum_sizes = np.cumsum(tamps_size)

    try:
        endpoint = cum_sizes.tolist().index(vector.shape[0])
    except ValueError:
        raise ValueError("Mismatch between vector size and tamps size")
    # NOTE: Special case for two-electron systems where T3 amplitudes are empty (zero-size)
    if mycc.do_diis_max_t: endpoint = (mycc.cc_order) * (mycc.cc_order + 3) // 2

    st = 0
    tamps = []
    for i in range(mycc.cc_order - 1):
        if st == endpoint:
            break
        t = []
        for na in range(i + 1, -1, -1):
            nb = i + 1 - na
            if na >= nb:
                nocc1, nocc2 = nocca, noccb
                nvir1, nvir2 = nvira, nvirb
                n1, n2 = na, nb
            else:
                nocc1, nocc2 = noccb, nocca
                nvir1, nvir2 = nvirb, nvira
                n1, n2 = nb, na
            t_ = np.zeros((nocc1,) * n1 + (nvir1,) * n1 + (nocc2,) * n2 + (nvir2,) * n2, dtype=vector.dtype)
            idx = mycc.unique_tamps_map[i][nb]
            t_[idx] = vector[cum_sizes[st] : cum_sizes[st + 1]].reshape(t_[idx].shape)
            restore_t_uhf_(t_, order=i + 1, pos=nb, do_tri=False)
            t.append(t_)
            st += 1
        tamps.append(t)

    if st < endpoint:
        t = []
        for na in range(mycc.cc_order, -1, -1):
            nb = mycc.cc_order - na
            if na >= nb:
                nocc1, nocc2 = nocca, noccb
                nvir1, nvir2 = nvira, nvirb
                n1, n2 = na, nb
            else:
                nocc1, nocc2 = noccb, nocca
                nvir1, nvir2 = nvirb, nvira
                n1, n2 = nb, na
            if mycc.do_tri_max_t:
                if n2 == 0:
                    xshape = (nx(nocc1, n1),) + (nx(nvir1, n1),)
                else:
                    xshape = (nx(nocc1, n1),) + (nx(nvir1, n1),) + (nx(nocc2, n2),) + (nx(nvir2, n2),)
                t_ = np.zeros(xshape, dtype=vector.dtype)
            else:
                t_ = np.zeros((nocc1,) * n1 + (nvir1,) * n1 + (nocc2,) * n2 + (nvir2,) * n2, dtype=vector.dtype)
            idx = mycc.unique_tamps_map[mycc.cc_order - 1][nb]
            t_[idx] = vector[cum_sizes[st] : cum_sizes[st + 1]].reshape(t_[idx].shape)
            restore_t_uhf_(t_, order=mycc.cc_order, pos=nb, do_tri=mycc.do_tri_max_t)
            t.append(t_)
            st += 1
        tamps.append(t)
    return tamps

def restore_t_uhf_(t, order=1, pos=0, do_tri=False):
    import itertools
    def permutation_sign(p):
        inv_count = sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
        return 1 if inv_count % 2 == 0 else -1

    na, nb = order - pos, pos
    if do_tri:
        return t
    else:
        tt = np.zeros_like(t)
        if na >= nb:
            n1, n2 = na, nb
        else:
            n1, n2 = nb, na

        if n1 >= 2:
            perms = list(itertools.permutations(range(n1)))
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*perm, *range(n1, 2 * n1), *range(2 * n1, 2 * n1 + 2 * n2))
                tt += sign * t.transpose(msg)
            t[:] = 0.0
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*range(n1), *[p + n1 for p in perm], *range(2 * n1, 2 * (n1 + n2)))
                t += sign * tt.transpose(msg)
        if n2 >= 2:
            perms = list(itertools.permutations(range(n2)))
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*range(2 * n1), *[p + 2 * n1 for p in perm], *range(2 * n1 + n2, 2 * (n1 + n2)))
                tt += sign * t.transpose(msg)
            t[:] = 0.0
            for idx, perm in enumerate(perms):
                sign = permutation_sign(perm)
                msg = (*range(2 * n1), *range(2 * n1, 2 * n1 + n2), *[p + 2 * n1 + n2 for p in perm])
                t += sign * tt.transpose(msg)

def _ao2mo_ucc(mycc, mo_coeff=None):
    if mycc._scf._eri is not None:
        logger.note(mycc, '_make_eris_incore_' + mycc.__class__.__name__)
        return _make_eris_incore_ucc(mycc, mo_coeff)
    elif getattr(mycc._scf, 'with_df', None):
        logger.note(mycc, '_make_df_eris_incore_' + mycc.__class__.__name__)
        return _make_df_eris_incore_ucc(mycc, mo_coeff)
    else:
        # NOTE: Handle the special case of a single-electron system
        logger.note(mycc, '_make_empty_eris_' + mycc.__class__.__name__)
        return _make_empty_eris_ucc(mycc, mo_coeff)

def restore_from_diis_(mycc, diis_file, inplace=True):
    '''Reuse an existed DIIS object in the CC calculation (UHF case).

    The CC amplitudes will be restored from the DIIS object. The `tamps` of the CC object will be overwritten
    by the generated `tamps`. The amplitudes vector and error vector will be reused in the CC calculation.
    '''
    from math import prod, factorial
    nx = lambda n, order: prod(n - i for i in range(order)) // factorial(order)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    adiis.restore(diis_file, inplace=inplace)

    ccvec = adiis.extrapolate()
    tamps = mycc.vector_to_amplitudes(ccvec)
    if mycc.do_diis_max_t:
        mycc.tamps = tamps
    else:
        mycc.tamps[:mycc.cc_order - 1] = tamps
        t = []
        for na in range(mycc.cc_order, -1, -1):
            nb = mycc.cc_order - na
            if na >= nb:
                n1, nocc1, nvir1, n2, nocc2, nvir2 = na, nocca, nvira, nb, noccb, nvirb
            else:
                n1, nocc1, nvir1, n2, nocc2, nvir2 = nb, noccb, nvirb, na, nocca, nvira
            if mycc.do_tri_max_t:
                if n2 >= 0:
                    shape = (nx(nocc1, n1),) + (nx(nvir1, n1),) + (nx(nocc2, n2),) + (nx(nvir2, n2),)
                else:
                    shape = (nx(nocc1, n1),) + (nx(nvir1, n1),)
            else:
                shape = (nocc1,) * (n1) + (nvir1,) * (n1) + (nocc2,) * (n2) + (nvir2,) * (n2)
            t_ = np.zeros(shape, dtype=ccvec.dtype)
            t.append(t_)
        mycc.tamps[-1] = t
    if inplace:
        mycc.diis = adiis
    return mycc

def vector_size_uhf(mycc, nmo=None, nocc=None):
    from math import prod, factorial
    nx = lambda nocc, order: prod(nocc - i for i in range(order)) // factorial(order)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    tamps_size = [0]
    # TODO: Should this function take `do_diis_max_t` into account?
    for i in range(mycc.cc_order):
        for na in range(i + 1, -1, -1):
            nb = i + 1 - na
            if na >= nb:
                tamps_size.append(nx(nocca, na) * nx(nvira, na) * nx(noccb, nb) * nx(nvirb, nb))
            else:
                tamps_size.append(nx(noccb, nb) * nx(nvirb, nb) * nx(nocca, na) * nx(nvira, na))
    cum_sizes = np.cumsum(tamps_size)
    return cum_sizes[-1]

def memory_estimate_log_uccsdt(mycc):
    '''Estimate the memory cost.'''
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    log.info('Approximate memory usage estimate')
    if mycc.do_tri_max_t:
        nocca3 = nocca * (nocca - 1) * (nocca - 2) // 6
        nvira3 = nvira * (nvira - 1) * (nvira - 2) // 6
        noccb3 = noccb * (noccb - 1) * (noccb - 2) // 6
        nvirb3 = nvirb * (nvirb - 1) * (nvirb - 2) // 6
        nocca2 = nocca * (nocca - 1) // 2
        nvira2 = nvira * (nvira - 1) // 2
        noccb2 = noccb * (noccb - 1) // 2
        nvirb2 = nvirb * (nvirb - 1) // 2
        t3_memory = (nocca3 * nvira3 + nocca2 * nvira2 * noccb * nvirb +
                    nocca * nvira * noccb2 * nvirb2 + noccb3 * nvirb3) * 8
        max_t3_memory = max(nocca3 * nvira3, nocca2 * nvira2 * noccb * nvirb,
                            nocca * nvira * noccb2 * nvirb2, noccb3 * nvirb3) * 8
    else:
        t3_memory = (nocca**3 * nvira**3 + nocca**2 * nvira**2 * noccb * nvirb +
                    nocca * nvira * noccb**2 * nvirb**2 + noccb**3 * nvirb**3) * 8
        max_t3_memory = max(nocca**3 * nvira**3, nocca**2 * nvira**2 * noccb * nvirb,
                    nocca * nvira * noccb**2 * nvirb**2, noccb**3 * nvirb**3) * 8
    log.info('    T3 memory               %8s', format_size(t3_memory))
    log.info('    R3 memory               %8s', format_size(t3_memory))
    if not mycc.do_tri_max_t:
        log.info('    Symmetrized T3 memory   %8s', format_size(max_t3_memory))
        if mycc.einsum_backend in ['numpy', 'pyscf']:
            log.info("    T3 einsum buffer        %8s", format_size(max_t3_memory))
    eris_memory = (nmoa**4 + nmoa**2 * nmob**2 + nmob**4) * 8
    log.info('    ERIs memory             %8s', format_size(eris_memory))
    log.info('    T1-ERIs memory          %8s', format_size(eris_memory))
    log.info('    Intermediates memory    %8s', format_size(eris_memory))
    if mycc.do_tri_max_t:
        blk_memory = mycc.blksize_o_aab * mycc.blksize_v_aab * max(nocca, noccb)**2 * max(nvira, nvirb)**2 * 8
        log.info("    Block workspace         %8s", format_size(blk_memory))
    if mycc.incore_complete:
        if mycc.do_diis_max_t:
            diis_memory = ((nocca * (nocca - 1) * (nocca - 2) // 6 * nvira * (nvira - 1) * (nvira - 2) // 6 +
                            nocca * (nocca - 1)// 2 * nvira * (nvira - 1) // 2 * noccb * nvirb +
                            nocca * nvira * noccb * (noccb - 1) // 2 * nvirb * (nvirb - 1) // 2 +
                            noccb * (noccb - 1) * (noccb - 2) // 6 * nvirb * (nvirb - 1) * (nvirb - 2) // 6)
                            * 8 * mycc.diis_space * 2)
        else:
            diis_memory = (nocca * (nocca - 1) // 2 * nvira * (nvira - 1) // 2
                        + nocca * nvira * noccb * nvirb
                        + noccb * (nvirb - 1) // 2 * nvirb * (nvirb - 1) // 2) * 8 * mycc.diis_space * 2
        log.info('    DIIS memory             %8s', format_size(diis_memory))
    else:
        diis_memory = 0.0
    if mycc.do_tri_max_t:
        total_memory = 2 * t3_memory + 3 * eris_memory + diis_memory + blk_memory
    else:
        total_memory = 3 * t3_memory + 3 * eris_memory + diis_memory
    if mycc.einsum_backend in ['numpy', 'pyscf']:
        total_memory += t3_memory
    log.info("Total estimated memory      %8s", format_size(total_memory))
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
            logger.warn(mycc, 'Consider using %s in `pyscf.cc.uccsdt` which stores triangular T amplitudes',
                        mycc.__class__.__name__)
        else:
            logger.warn(mycc, 'Consider reducing `blksize_o_aaa`, `blksize_v_aaa`, `blksize_o_aab`, '
                                'and `blksize_v_aab` to reduce memory usage')
    return mycc

def build_unique_tamps_map_uhf(mycc):
    '''Build the mapping for the symmetry-unique part of the T-amplitudes.'''
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    unique_tamps_map = []
    # t1
    t1a_idx = (slice(None), slice(None))
    t1b_idx = (slice(None), slice(None))
    unique_tamps_map.append([t1a_idx, t1b_idx])
    # t2
    def compute_t2_idx_uhf(nocc, nvir):
        i_idx, j_idx = np.triu_indices(nocc, k=1)
        a_idx, b_idx = np.triu_indices(nvir, k=1)
        I = np.repeat(i_idx, len(a_idx))
        J = np.repeat(j_idx, len(a_idx))
        A = np.tile(a_idx, len(i_idx))
        B = np.tile(b_idx, len(i_idx))
        t2_idx = (I, J, A, B)
        return t2_idx

    t2aa_idx = compute_t2_idx_uhf(nocca, nvira)
    t2ab_idx = (slice(None), slice(None), slice(None), slice(None))
    t2bb_idx = compute_t2_idx_uhf(noccb, nvirb)
    unique_tamps_map.append([t2aa_idx, t2ab_idx, t2bb_idx])
    # t3
    if mycc.do_diis_max_t:
        if mycc.do_tri_max_t:
            t3aaa_idx = (slice(None), slice(None))
            t3aab_idx = (slice(None), slice(None), slice(None), slice(None))
            t3bba_idx = (slice(None), slice(None), slice(None), slice(None))
            t3bbb_idx = (slice(None), slice(None))
            unique_tamps_map.append([t3aaa_idx, t3aab_idx, t3bba_idx, t3bbb_idx])
        else:
            def build_t3aaa_indices_uhf(nocc, nvir):
                ii, jj, kk = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
                i_idx, j_idx, k_idx = np.where((ii < jj) & (jj < kk))
                aa, bb, cc = np.meshgrid(np.arange(nvir), np.arange(nvir), np.arange(nvir), indexing='ij')
                a_idx, b_idx, c_idx = np.where((aa < bb) & (bb < cc))
                I = np.repeat(i_idx, len(a_idx))
                J = np.repeat(j_idx, len(a_idx))
                K = np.repeat(k_idx, len(a_idx))
                A = np.tile(a_idx, len(i_idx))
                B = np.tile(b_idx, len(i_idx))
                C = np.tile(c_idx, len(i_idx))
                t3aaa_idx = (I, J, K, A, B, C)
                return t3aaa_idx

            def build_t3aab_indices_uhf(nocca, nvira, noccb, nvirb):
                i_idx, j_idx = np.triu_indices(nocca, k=1)
                a_idx, b_idx = np.triu_indices(nvira, k=1)
                I = np.repeat(i_idx, len(a_idx))
                J = np.repeat(j_idx, len(a_idx))
                A = np.tile(a_idx, len(i_idx))
                B = np.tile(b_idx, len(i_idx))
                t3aab_idx = (I, J, A, B, slice(None), slice(None))
                return t3aab_idx

            t3aaa_idx = build_t3aaa_indices_uhf(nocca, nvira)
            t3aab_idx = build_t3aab_indices_uhf(nocca, nvira, noccb, nvirb)
            t3bba_idx = build_t3aab_indices_uhf(noccb, nvirb, nocca, nvira)
            t3bbb_idx = build_t3aaa_indices_uhf(noccb, nvirb)
            unique_tamps_map.append([t3aaa_idx, t3aab_idx, t3bba_idx, t3bbb_idx])
    return unique_tamps_map

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
        lib.chkfile.save(mycc.chkfile, 'uccsdt', cc_chk)
    else:
        lib.chkfile.save(mycc.chkfile, 'uccsdt_highm', cc_chk)

def tamps_tri2full_uhf(mycc, tamps_tri):
    '''Convert triangular-stored T amplitudes to their full tensor form (UHF case).'''
    assert mycc.cc_order in (3,), "`cc_order` must be 3"
    if mycc.cc_order == 3:
        assert len(tamps_tri) == 4, "`tamps_tri` must contain (t3aaa, t3aab, t3bba, t3bbb)"
        assert tamps_tri[0].ndim == 2, "t3aaa must be 2-dimensional"
        assert tamps_tri[1].ndim == 4, "t3aab must be 4-dimensional"
        assert tamps_tri[2].ndim == 4, "t3bba must be 4-dimensional"
        assert tamps_tri[3].ndim == 2, "t3bbb must be 2-dimensional"

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    t3aaa_full = np.zeros((nocca,) * 3 + (nvira,) * 3, dtype=tamps_tri[0].dtype)
    _unp_aaa_(mycc, tamps_tri[0], t3aaa_full, 0, nocca, 0, nocca, 0, nocca, 0, nvira, 0, nvira, 0, nvira,
                blk_i=nocca, blk_j=nocca, blk_k=nocca, blk_a=nvira, blk_b=nvira, blk_c=nvira)

    t3aab_full = np.zeros((nocca,) * 2 + (nvira,) * 2 + (noccb,) + (nvirb,), dtype=tamps_tri[1].dtype)
    _unp_aab_(mycc, tamps_tri[1], t3aab_full, 0, nocca, 0, nocca, 0, nvira, 0, nvira,
                blk_i=nocca, blk_j=nocca, blk_a=nvira, blk_b=nvira, dim4=noccb, dim5=nvirb)

    t3bba_full = np.zeros((noccb,) * 2 + (nvirb,) * 2 + (nocca,) + (nvira,), dtype=tamps_tri[2].dtype)
    _unp_bba_(mycc, tamps_tri[2], t3bba_full, 0, noccb, 0, noccb, 0, nvirb, 0, nvirb,
                blk_i=noccb, blk_j=noccb, blk_a=nvirb, blk_b=nvirb, dim4=nocca, dim5=nvira)

    t3bbb_full = np.zeros((noccb,) * 3 + (nvirb,) * 3, dtype=tamps_tri[3].dtype)
    _unp_bbb_(mycc, tamps_tri[3], t3bbb_full, 0, noccb, 0, noccb, 0, noccb, 0, nvirb, 0, nvirb, 0, nvirb,
                blk_i=noccb, blk_j=noccb, blk_k=noccb, blk_a=nvirb, blk_b=nvirb, blk_c=nvirb)
    return t3aaa_full, t3aab_full, t3bba_full, t3bbb_full

def tamps_full2tri_uhf(mycc, tamps_full):
    '''Convert full T amplitudes to their triangular-stored form (UHF case).'''
    # TODO: Generalize this function to T amplitudes of arbitrary order
    assert mycc.cc_order in (3,), "`cc_order` must be 3"
    assert len(tamps_full) - 1 == 3, "`tamps_full` must contain (t3aaa, t3aab, t3bba, t3bbb)"

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    t3aaa, t3aab, t3bba, t3bbb = tamps_full

    i, j, k = np.meshgrid(np.arange(nocca), np.arange(nocca), np.arange(nocca), indexing='ij')
    t3_o_map = np.where((i < j) & (j < k))
    a, b, c = np.meshgrid(np.arange(nvira), np.arange(nvira), np.arange(nvira), indexing='ij')
    t3_v_map = np.where((a < b) & (b < c))
    t3aaa_tri = t3aaa[t3_o_map[0][:, None], t3_o_map[1][:, None], t3_o_map[2][:, None],
                        t3_v_map[0][None, :], t3_v_map[1][None, :], t3_v_map[2][None, :]]

    i, j = np.meshgrid(np.arange(nocca), np.arange(nocca), indexing='ij')
    t3_o_map = np.where(i < j)
    a, b = np.meshgrid(np.arange(nvira), np.arange(nvira), indexing='ij')
    t3_v_map = np.where(a < b)
    t3aab_tri = t3aab[t3_o_map[0][:, None], t3_o_map[1][:, None], t3_v_map[0][None, :], t3_v_map[1][None, :], :, :]

    i, j = np.meshgrid(np.arange(noccb), np.arange(noccb), indexing='ij')
    t3_o_map = np.where(i < j)
    a, b = np.meshgrid(np.arange(nvirb), np.arange(nvirb), indexing='ij')
    t3_v_map = np.where(a < b)
    t3bba_tri = t3bba[t3_o_map[0][:, None], t3_o_map[1][:, None], t3_v_map[0][None, :], t3_v_map[1][None, :], :, :]

    i, j, k = np.meshgrid(np.arange(noccb), np.arange(noccb), np.arange(noccb), indexing='ij')
    t3_o_map = np.where((i < j) & (j < k))
    a, b, c = np.meshgrid(np.arange(nvirb), np.arange(nvirb), np.arange(nvirb), indexing='ij')
    t3_v_map = np.where((a < b) & (b < c))
    t3bbb_tri = t3bbb[t3_o_map[0][:, None], t3_o_map[1][:, None], t3_o_map[2][:, None],
                        t3_v_map[0][None, :], t3_v_map[1][None, :], t3_v_map[2][None, :]]
    return t3aaa_tri, t3aab_tri, t3bba_tri, t3bbb_tri

def tamps_rhf2uhf(mycc, tamps_rhf):
    '''Convert T amplitudes from an RCCSDT calculation to a UCCSDT representation.
    This function operates only on full T amplitudes. For triangular-stored amplitudes,
    first reconstruct the full tensor, then apply this conversion.
    '''
    order = len(tamps_rhf)
    assert order in (1, 2, 3), "Only T1, T2, and T3 amplitudes transformations are supported"
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    assert (nocca == noccb) and (nmoa == nmob), "This function only works for spin-restricted systems"
    tamps_uhf = []
    if order >= 1:
        tamps_uhf.append([tamps_rhf[0].copy(), tamps_rhf[0].copy()])
    if order >= 2:
        t2_uhf = [None, None, None]
        t2_rhf = tamps_rhf[1]
        t2_uhf[0] = t2_rhf - t2_rhf.transpose(0, 1, 3, 2)
        # NOTE: t2ab is store as (nocca, nvira, noccb, nvirb), which is different from pyscf.cc.uccsd
        t2_uhf[1] = t2_rhf.transpose(0, 2, 1, 3).copy()
        t2_uhf[2] = t2_uhf[0].copy()
        tamps_uhf.append(t2_uhf)
    if order >= 3:
        t3_uhf = [None, None, None, None]
        t3_rhf = tamps_rhf[2]
        t3_uhf[0] = (t3_rhf - t3_rhf.transpose(0, 1, 2, 4, 3, 5) - t3_rhf.transpose(0, 1, 2, 5, 4, 3)
                    - t3_rhf.transpose(0, 1, 2, 3, 5, 4) + t3_rhf.transpose(0, 1, 2, 4, 5, 3)
                    + t3_rhf.transpose(0, 1, 2, 5, 3, 4))
        # NOTE: t3aab is store as (nocca, nocca, nvira, nvira, noccb, nvirb)
        t3_uhf[1] = t3_rhf - t3_rhf.transpose(0, 1, 2, 4, 3, 5)
        t3_uhf[1] = t3_uhf[1].transpose(0, 1, 3, 4, 2, 5)
        t3_uhf[2] = t3_uhf[1].copy()
        t3_uhf[3] = t3_uhf[0].copy()
        tamps_uhf.append(t3_uhf)
    return tamps_uhf

def tamps_uhf2rhf(mycc, tamps_uhf):
    '''Convert T amplitudes from a UCCSDT calculation to an RCCSDT representation.
    This function operates only on full T amplitudes. For triangular-stored amplitudes,
    first reconstruct the full tensor, then apply this conversion.
    '''
    order = len(tamps_uhf)
    assert order in (1, 2, 3), "Only T1, T2, and T3 amplitude transformations are supported"
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    assert (nocca == noccb) and (nmoa == nmob), "This function only works for spin-restricted systems"
    tamps_rhf = []
    if order >= 1:
        t1a, t1b = tamps_uhf[0]
        assert np.max(np.abs(t1a - t1b)) < 1e-8, "The spin symmetry of `tamps_uhf` may not be preserved"
        tamps_rhf.append(t1a)
    if order >= 2:
        t2aa, t2ab, t2bb = tamps_uhf[1]
        assert np.max(np.abs(t2aa - t2bb)) < 1e-8, "The spin symmetry of `tamps_uhf` may not be preserved"
        # NOTE: t2ab is store as (nocca, nvira, noccb, nvirb), which is different from pyscf.cc.uccsd
        tamps_rhf.append(t2ab.transpose(0, 2, 1, 3))
    if order >= 3:
        t3aaa, t3aab, t3bba, t3bbb = tamps_uhf[2]
        assert np.max(np.abs(t3aaa - t3bbb)) < 1e-8, "The spin symmetry of `tamps_uhf` may not be preserved"
        assert np.max(np.abs(t3aab - t3bba)) < 1e-8, "The spin symmetry of `tamps_uhf` may not be preserved"
        # Use the previously imposed constraint on T3: the fully symmetrized component over (a, b, c) is zero, i.e.,
        # T_{ijk}^{abc} + T_{ijk}^{acb} + T_{ijk}^{bac} + T_{ijk}^{bca} + T_{ijk}^{cab} + T_{ijk}^{cba} = 0
        # NOTE: t3aab is store as (nocca, nocca, nvira, nvira, noccb, nvirb)
        t3_rhf = (-1.0 / 6.0) * t3aaa + (1.0 / 3.0) * (t3aab.transpose(0, 1, 4, 2, 3, 5)
                    + t3aab.transpose(0, 4, 1, 2, 5, 3) + t3aab.transpose(4, 0, 1, 5, 2, 3))
        tamps_rhf.append(t3_rhf)
    return tamps_rhf


class UCCSDT(ccsd.CCSDBase):

    conv_tol = getattr(__config__, 'cc_uccsdt_UCCSDT_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_uccsdt_UCCSDT_conv_tol_normt', 1e-6)
    cc_order = getattr(__config__, 'cc_uccsdt_UCCSDT_cc_order', 3)
    do_diis_max_t = getattr(__config__, 'cc_uccsdt_UCCSDT_do_diis_max_t', True)
    blksize_o_aaa = getattr(__config__, 'cc_uccsdt_UCCSDT_blksize_o_aaa', 8)
    blksize_v_aaa = getattr(__config__, 'cc_uccsdt_UCCSDT_blksize_v_aaa', 64)
    blksize_o_aab = getattr(__config__, 'cc_uccsdt_UCCSDT_blksize_o_aab', 8)
    blksize_v_aab = getattr(__config__, 'cc_uccsdt_UCCSDT_blksize_v_aab', 64)
    einsum_backend = getattr(__config__, 'cc_uccsdt_UCCSDT_einsum_backend', 'numpy')

    _keys = {
        'max_cycle', 'conv_tol', 'iterative_damping', 'conv_tol_normt', 'diis', 'diis_space', 'diis_file',
        'diis_start_cycle', 'diis_start_energy_diff', 'async_io', 'incore_complete', 'callback',
        'mol', 'verbose', 'stdout', 'frozen', 'level_shift', 'mo_coeff', 'mo_occ', 'cycles', 'emp2', 'e_hf',
        'converged', 'e_corr', 'chkfile', 'cc_order', 'do_diis_max_t', 'blksize_o_aaa', 'blksize_v_aaa',
        'blksize_o_aab', 'blksize_v_aab', 'einsum_backend', 'tamps', 'unique_tamps_map', 't2c_map_6f_oa',
        't2c_mask_6f_oa', 't2c_map_2f_oa', 't2c_mask_2f_oa', 't2c_map_6f_va', 't2c_mask_6f_va', 't2c_map_2f_va',
        't2c_mask_2f_va', 't2c_map_6f_ob', 't2c_mask_6f_ob', 't2c_map_2f_ob', 't2c_mask_2f_ob', 't2c_map_6f_vb',
        't2c_mask_6f_vb', 't2c_map_2f_vb', 't2c_mask_2f_vb', 'unique_tamps_map'
    }

    @property
    def t1(self):
        return self.tamps[0]

    @t1.setter
    def t1(self, val):
        self.tamps[0] = val

    @property
    def t2(self):
        return self.tamps[1]

    @t2.setter
    def t2(self, val):
        self.tamps[1] = val

    @property
    def t3(self):
        return self.tamps[2]

    @t3.setter
    def t3(self, val):
        self.tamps[2] = val

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        self.tamps = [None, None, None]
        ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.unique_tamps_map = None

    do_tri_max_t = property(lambda self: True)

    def set_einsum_backend(self, backend):
        self.einsum_backend = backend

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf
    ao2mo = _ao2mo_ucc
    energy = energy_uhf
    init_amps = init_amps_uhf
    restore_from_diis_ = restore_from_diis_
    memory_estimate_log = memory_estimate_log_uccsdt
    update_amps_ = update_amps_uccsdt_tri_
    amplitudes_to_vector = amplitudes_to_vector_uhf
    vector_to_amplitudes = vector_to_amplitudes_uhf
    build_unique_tamps_map = build_unique_tamps_map_uhf
    setup_tri2block = setup_tri2block_t3_uhf
    vector_size = vector_size_uhf
    run_diis = run_diis
    _finalize = _finalize
    dump_flags = dump_flags
    dump_chk = dump_chk
    tamps_tri2full = tamps_tri2full_uhf
    tamps_full2tri = tamps_full2tri_uhf
    tamps_rhf2uhf = tamps_rhf2uhf
    tamps_uhf2rhf = tamps_uhf2rhf

    def kernel(self, tamps=None, eris=None):
        return self.ccsdt(tamps, eris)

    def ccsdt(self, tamps=None, eris=None):
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

        nocca, noccb = eris.nocc
        nmoa, nmob = eris.fock[0].shape[0], eris.fock[1].shape[0]
        nvira, nvirb = nmoa - nocca, nmob - noccb

        if self.do_tri_max_t:
            (self.t2c_map_6f_oa, self.t2c_mask_6f_oa, self.t2c_map_2f_oa, self.t2c_mask_2f_oa,
            self.t2c_map_6f_va, self.t2c_mask_6f_va, self.t2c_map_2f_va, self.t2c_mask_2f_va,
            self.t2c_map_6f_ob, self.t2c_mask_6f_ob, self.t2c_map_2f_ob, self.t2c_mask_2f_ob,
            self.t2c_map_6f_vb, self.t2c_mask_6f_vb, self.t2c_map_2f_vb, self.t2c_mask_2f_vb) = self.setup_tri2block()

            self.blksize_o_aaa = min(self.blksize_o_aaa, max(nocca, noccb))
            self.blksize_v_aaa = min(self.blksize_v_aaa, max(nvira, nvirb))
            self.blksize_o_aab = min(self.blksize_o_aab, max(nocca, noccb))
            self.blksize_v_aab = min(self.blksize_v_aab, max(nvira, nvirb))
            log.info('blksize_o_aaa %5d    blksize_v_aaa %5d'%(self.blksize_o_aaa, self.blksize_v_aaa))
            log.info('blksize_o_aab %5d    blksize_v_aab %5d'%(self.blksize_o_aab, self.blksize_v_aab))

            if self.blksize_v_aaa > (max(nvira, nvirb) + 1) // 2:
                logger.warn(self, 'A large `blksize_v_aaa` is being used, which may cause large memory consumption\n'
                            '      for storing contraction intermediates. If memory is sufficient, consider using\n'
                            '      `pyscf.cc.uccsdt_highm.UCCSDT` instead.')
            if self.blksize_v_aab > (max(nvira, nvirb) + 1) // 2:
                logger.warn(self, 'A large `blksize_v_aab` is being used, which may cause large memory consumption\n'
                            '      for storing contraction intermediates. If memory is sufficient, consider using\n'
                            '      `pyscf.cc.uccsdt_highm.UCCSDT` instead.')

        self.memory_estimate_log()
        self.unique_tamps_map = self.build_unique_tamps_map()

        self.converged, self.e_corr, self.tamps = kernel(self, eris, tamps, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt, verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.tamps

    def ccsdt_q(self, tamps, eris=None):
        raise NotImplementedError


class _PhysicistsERIs:
    '''<pq|rs> = (pr|qs). Without antisymmetrization'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = mycc.get_frozen_mask()
        self.mo_coeff = mo_coeff = (mo_coeff[0][:, mo_idx[0]], mo_coeff[1][:, mo_idx[1]])

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.focka = reduce(np.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(np.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
        self.fock = (self.focka, self.fockb)

        nocca, noccb = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
        gap_a = abs(mo_ea[:nocca, None] - mo_ea[None, nocca:])
        gap_b = abs(mo_eb[:noccb, None] - mo_eb[None, noccb:])
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for %s', gap_a, gap_b, mycc.__class__.__name__)
        return self

def _make_eris_incore_ucc(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa, moa, mob, mob))
    else:
        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
        eri_ab = ao2mo.general(mycc._scf._eri, (moa, moa, mob, mob), compact=False)

    eri_aa = eri_aa.reshape(nmoa, nmoa, nmoa, nmoa)
    eri_ab = eri_ab.reshape(nmoa, nmoa, nmob, nmob)
    eri_bb = eri_bb.reshape(nmob, nmob, nmob, nmob)

    eris.pppp = eri_aa.transpose(0, 2, 1, 3)
    eris.PPPP = eri_bb.transpose(0, 2, 1, 3)
    eris.pPpP = eri_ab.transpose(0, 2, 1, 3)

    if not eris.pppp.flags['C_CONTIGUOUS']:
        eris.pppp = np.ascontiguousarray(eris.pppp)
    if not eris.PPPP.flags['C_CONTIGUOUS']:
        eris.PPPP = np.ascontiguousarray(eris.PPPP)
    if not eris.pPpP.flags['C_CONTIGUOUS']:
        eris.pPpP = np.ascontiguousarray(eris.pPpP)

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris

def _make_df_eris_incore_ucc(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    moa, mob = eris.mo_coeff
    nocca, noccb = eris.nocc
    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    naux = mycc._scf.with_df.get_naoaux()

    # --- Three-center integrals
    # (L|aa)
    Lpq = numpy.empty((naux, nmoa, nmoa))
    # (L|bb)
    LPQ = numpy.empty((naux, nmob, nmob))
    p1 = 0
    # Transform three-center integrals to MO basis
    einsum = lib.einsum
    Lpq_tmp = None
    for eri1 in mycc._scf.with_df.loop():
        eri1 = lib.unpack_tril(eri1).reshape(-1, nao, nao)
        # (L|aa)
        Lpq_tmp = einsum('Lab,ap,bq->Lpq', eri1, moa, moa)
        p0, p1 = p1, p1 + Lpq_tmp.shape[0]
        Lpq[p0:p1, :, :] = Lpq_tmp[:, :, :]
        Lpq_tmp = None
        # (L|bb)
        Lpq_tmp = einsum('Lab,ap,bq->Lpq', eri1, mob, mob)
        LPQ[p0:p1, :, :] = Lpq_tmp[:, :, :]
        Lpq_tmp = None
    Lpq = Lpq.reshape(naux, nmoa * nmoa)
    LPQ = LPQ.reshape(naux, nmob * nmob)

    # --- Four-center integrals
    # <aa|aa>
    eris.pppp = lib.ddot(Lpq.T, Lpq).reshape(nmoa, nmoa, nmoa, nmoa).transpose(0, 2, 1, 3)
    if not eris.pppp.flags['C_CONTIGUOUS']:
        eris.pppp = np.ascontiguousarray(eris.pppp)
    # <bb|bb>
    eris.PPPP = lib.ddot(LPQ.T, LPQ).reshape(nmob, nmob, nmob, nmob).transpose(0, 2, 1, 3)
    if not eris.PPPP.flags['C_CONTIGUOUS']:
        eris.PPPP = np.ascontiguousarray(eris.PPPP)
    # <ab|ab>
    eris.pPpP = lib.ddot(Lpq.T, LPQ).reshape(nmoa, nmoa, nmob, nmob).transpose(0, 2, 1, 3)
    if not eris.pPpP.flags['C_CONTIGUOUS']:
        eris.pPpP = np.ascontiguousarray(eris.pPpP)

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris

def _make_empty_eris_ucc(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    from pyscf.scf.uhf import UHF
    assert isinstance(mycc._scf, UHF)
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    eris.pppp = np.zeros((nmoa, nmoa, nmoa, nmoa), dtype=mycc.mo_coeff.dtype)
    eris.PPPP = np.zeros((nmob, nmob, nmob, nmob), dtype=mycc.mo_coeff.dtype)
    eris.pPpP = np.zeros((nmoa, nmob, nmoa, nmob), dtype=mycc.mo_coeff.dtype)

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris

class _IMDS:

    def __init__(self):
        self.t1_fock = None
        self.t1_eris = None
        self.F_oo, self.F_OO = None, None
        self.F_vv, self.F_VV = None, None
        self.W_oooo, self.W_oOoO, self.W_OOOO = None, None, None
        self.W_ovoo, self.W_oVoO, self.W_OVOO = None, None, None
        self.W_vOoO, self.W_oVoV, self.W_vOvO, self.W_vVoV = None, None, None, None
        self.W_voov, self.W_vOoV, self.W_VoOv, self.W_VOOV = None, None, None, None
        self.W_vvvo, self.W_vVvO, self.W_VVVO = None, None, None
        self.W_vvvv, self.W_vVvV, self.W_VVVV = None, None, None


if __name__ == "__main__":

    from pyscf import gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 -0.757, 0.587; H 0 0.757, 0.587", basis='631g', verbose=3, spin=2)
    mf = scf.UHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()
    ref_ecorr = -0.1092563391390793
    mycc = UCCSDT(mf, frozen=0)
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

    # comparison with the high-memory version
    from pyscf.cc.uccsdt_highm import UCCSDT as UCCSDThm
    mycc2 = UCCSDThm(mf, frozen=0)
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

    t3_tri = mycc.tamps[2]
    t3_full = mycc2.tamps[2]
    t3_tri_from_full = mycc2.tamps_full2tri(t3_full)
    t3_full_from_tri = mycc.tamps_tri2full(t3_tri)

    print('energy difference                             % .10e' % (mycc.e_tot - mycc2.e_tot))
    print('max(abs(t1a difference))                      % .10e' % np.max(np.abs(mycc.t1[0] - mycc2.t1[0])))
    print('max(abs(t1b difference))                      % .10e' % np.max(np.abs(mycc.t1[1] - mycc2.t1[1])))
    print('max(abs(t2aa difference))                     % .10e' % np.max(np.abs(mycc.t2[0] - mycc2.t2[0])))
    print('max(abs(t2ab difference))                     % .10e' % np.max(np.abs(mycc.t2[1] - mycc2.t2[1])))
    print('max(abs(t2bb difference))                     % .10e' % np.max(np.abs(mycc.t2[2] - mycc2.t2[2])))
    print('max(abs(t3aaa_tri - t3aaa_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[0] - t3_tri_from_full[0])))
    print('max(abs(t3aaa_full - t3aaa_full_from_tri))    % .10e' % np.max(np.abs(t3_full[0] - t3_full_from_tri[0])))
    print('max(abs(t3aab_tri - t3aab_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[1] - t3_tri_from_full[1])))
    print('max(abs(t3aab_full - t3aab_full_from_tri))    % .10e' % np.max(np.abs(t3_full[1] - t3_full_from_tri[1])))
    print('max(abs(t3bba_tri - t3bba_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[2] - t3_tri_from_full[2])))
    print('max(abs(t3bba_full - t3bba_full_from_tri))    % .10e' % np.max(np.abs(t3_full[2] - t3_full_from_tri[2])))
    print('max(abs(t3bbb_tri - t3bbb_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[3] - t3_tri_from_full[3])))
    print('max(abs(t3bbb_full - t3bbb_full_from_tri))    % .10e' % np.max(np.abs(t3_full[3] - t3_full_from_tri[3])))
