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
RHF-CCSDT with T3 amplitudes stored only for the i <= j <= k index combinations.
T1-dressed formalism is used, where the T1 amplitudes are absorbed into the Fock matrix and ERIs.

Ref:
J. Chem. Phys. 142, 064108 (2015); DOI:10.1063/1.4907278
Chem. Phys. Lett. 228, 233 (1994); DOI:10.1016/0009-2614(94)00898-1
'''

import numpy as np
import numpy
import functools
from functools import reduce
import ctypes
from pyscf import ao2mo, lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf.cc import ccsd, _ccsd
from pyscf import __config__


_libccsdt = lib.load_library('libccsdt')

def _einsum(einsum_backend, script, *tensors, out=None, alpha=1.0, beta=0.0):
    '''Wrapper for einsum supporting pytblis, pyscf.lib.einsum, or numpy.einsum backends.'''
    if einsum_backend == 'pytblis':
        try:
            import pytblis
        except ImportError:
            import numpy as np
            einsum_backend = 'numpy'
    elif einsum_backend == 'numpy':
        import numpy as np
    elif einsum_backend == 'pyscf':
        from pyscf import lib
    else:
        raise ValueError(f"Unknown einsum_backend: {einsum_backend}")

    def _fix_strides_for_pytblis(arr):
        '''Fix strides for pytblis'''
        import itertools, numpy as np
        if arr.strides == (0,) * arr.ndim:
            strides = tuple(itertools.accumulate([x if x != 0 else 1 for x in arr.shape],
                            lambda x, y: x * y, initial=1))[:-1]
            arr = np.lib.stride_tricks.as_strided(arr, strides=strides)
        return arr

    if einsum_backend == 'pytblis':
        if out is None:
            result = pytblis.contract(script, *tensors)
        else:
            tensors = [_fix_strides_for_pytblis(x) for x in tensors]
            out = _fix_strides_for_pytblis(out)
            pytblis.contract(script, *tensors, out=out, alpha=alpha, beta=beta)
            return
    elif einsum_backend == 'pyscf':
        result = lib.einsum(script, *tensors, optimize='optimal')
    else:
        result = np.einsum(script, *tensors, optimize='optimal')

    if out is None:
        if alpha != 1.0:
            result = alpha * result
        return result
    else:
        if beta == 0.0:
            out[:] = alpha * result
        else:
            out[:] = alpha * result + beta * out
        return out

def t3_spin_summation_inplace_(A, nocc3, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    pattern_c = pattern.encode('utf-8')
    _libccsdt.t3_spin_summation_inplace_(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc3), ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return A

def unpack_t3_tri2block_(t3, t3_blk, map_, mask, i0, i1, j0, j1, k0, k1, nocc, nvir, blk_i, blk_j, blk_k):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_ = np.ascontiguousarray(map_)
    mask = np.ascontiguousarray(mask)
    _libccsdt.unpack_t3_tri2block_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k)
    )
    return t3_blk

def unpack_t3_tri2single_pair_(t3, t3_blk, map_, mask, i0, j0, k0, nocc, nvir):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_ = np.ascontiguousarray(map_)
    mask = np.ascontiguousarray(mask)
    _libccsdt.unpack_t3_tri2single_pair_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(j0), ctypes.c_int64(k0),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
    )
    return t3_blk

def unpack_t3_tri2block_pair_(t3, t3_blk, map_, mask, i0, i1, j0, j1, k0, k1, nocc, nvir, blk_i, blk_j, blk_k):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64 and mask.dtype == np.bool_
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_ = np.ascontiguousarray(map_)
    mask = np.ascontiguousarray(mask)
    _libccsdt.unpack_t3_tri2block_pair_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k)
    )
    return t3_blk

def accumulate_t3_block2tri_(t3, t3_blk, map_, i0, i1, j0, j1, k0, k1, nocc, nvir, blk_i, blk_j, blk_k, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_ = np.ascontiguousarray(map_)
    _libccsdt.accumulate_t3_block2tri_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(i1),
        ctypes.c_int64(j0), ctypes.c_int64(j1),
        ctypes.c_int64(k0), ctypes.c_int64(k1),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_int64(blk_i), ctypes.c_int64(blk_j), ctypes.c_int64(blk_k),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def accumulate_t3_single2tri_(t3, t3_blk, map_, i0, j0, k0, nocc, nvir, alpha, beta):
    assert t3.dtype == np.float64 and t3_blk.dtype == np.float64
    assert map_.dtype == np.int64
    t3 = np.ascontiguousarray(t3)
    t3_blk = np.ascontiguousarray(t3_blk)
    map_ = np.ascontiguousarray(map_)
    _libccsdt.accumulate_t3_single2tri_(
        t3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t3_blk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        map_.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(i0), ctypes.c_int64(j0), ctypes.c_int64(k0),
        ctypes.c_int64(nocc), ctypes.c_int64(nvir),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return t3

def _unpack_t3_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, blksize0=None, blksize1=None, blksize2=None):
    '''Unpack triangular-stored T3 amplitudes into the block `t3_full[i0:i1, j0:j1, k0:k1, :, :, :]`'''
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    unpack_t3_tri2block_(t3, t3_blk, mycc.tri2block_map, mycc.tri2block_mask,
                        i0, i1, j0, j1, k0, k1, mycc.nocc, mycc.nmo - mycc.nocc, blksize0, blksize1, blksize2)
    return t3_blk

def _unpack_t3_s_pair_(mycc, t3, t3_blk, i0, j0, k0):
    '''Unpack triangular-stored T3 amplitudes into the block
    `t3_full[i0, j0, k0, :, :, :] + t3_full[j0, i0, k0, :, :, :].transpose(1, 0, 2)`
    '''
    unpack_t3_tri2single_pair_(t3, t3_blk, mycc.tri2block_map, mycc.tri2block_mask,
                                i0, j0, k0, mycc.nocc, mycc.nmo - mycc.nocc)
    return t3_blk

def _unpack_t3_pair_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1, blksize0=None, blksize1=None, blksize2=None):
    '''Unpack triangular-stored T3 amplitudes into the block
    `t3_full[i0:i1, j0:j1, k0:k1, :, :, :] + t3_full[k0:k1, j0:j1, i0:i1, :, :, :].transpose(0, 1, 2, 3, 5, 4)`
    '''
    if blksize0 is None: blksize0 = mycc.blksize_oovv
    if blksize1 is None: blksize1 = mycc.nocc
    if blksize2 is None: blksize2 = mycc.blksize_oovv
    unpack_t3_tri2block_pair_(t3, t3_blk, mycc.tri2block_map, mycc.tri2block_mask,
                                i0, i1, j0, j1, k0, k1, mycc.nocc, mycc.nmo - mycc.nocc, blksize0, blksize1, blksize2)
    return t3_blk

def _accumulate_t3_(mycc, t3, t3_blk, i0, i1, j0, j1, k0, k1,
                        blksize0=None, blksize1=None, blksize2=None, alpha=1.0, beta=0.0):
    if blksize0 is None: blksize0 = mycc.blksize
    if blksize1 is None: blksize1 = mycc.blksize
    if blksize2 is None: blksize2 = mycc.blksize
    accumulate_t3_block2tri_(t3, t3_blk, mycc.tri2block_map, i0, i1, j0, j1, k0, k1,
                        mycc.nocc, mycc.nmo - mycc.nocc, blksize0, blksize1, blksize2, alpha=alpha, beta=beta)
    return t3

def _accumulate_t3_s_(mycc, t3, t3_blk, i0, j0, k0, alpha=1.0, beta=0.0):
    accumulate_t3_single2tri_(t3, t3_blk, mycc.tri2block_map, i0, j0, k0,
                                mycc.nocc, mycc.nmo - mycc.nocc, alpha=alpha, beta=beta)
    return t3

def setup_tri2block_rhf(mycc):
    '''Build the map used to unpack and accumulate between the triangular-stored T3 and the block of full T3 tensor.'''
    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    cc_order = mycc.cc_order
    nocc = mycc.nocc
    noccp = nx(nocc, cc_order)
    nsymm = factorial(cc_order)

    tri2block_map = np.zeros((nsymm,) + (nocc,) * cc_order, dtype=np.int64)
    tri2block_mask = np.zeros((nsymm,) + (nocc,) * cc_order, dtype=np.bool_)
    tri2block_tp = []

    idx = np.meshgrid(*[np.arange(nocc)] * cc_order, indexing='ij')
    idx = np.stack(idx)
    tamps_map = np.where(np.all(np.diff(idx, axis=0) >= 0, axis=0))

    import itertools
    perms = list(itertools.permutations(range(cc_order)))
    for i, perm in enumerate(perms):
        inds = tuple(tamps_map[p] for p in perm)
        tri2block_map[(i,) + inds] = np.arange(noccp)

    labels = tuple("ijklmnop"[:cc_order])
    collect_relation = set(itertools.combinations(labels, 2))
    var_map = dict(zip(labels, idx))
    for idx, perm in enumerate(perms):
        indices = np.argsort(perm)
        vars_sorted = [var_map[labels[indices[i]]] for i in range(cc_order)]
        comparisons = []
        for comparison_idx in range(cc_order - 1):
            left_label = labels[indices[comparison_idx]]
            right_label = labels[indices[comparison_idx + 1]]
            if (right_label, left_label) in collect_relation:
                comparisons.append(vars_sorted[comparison_idx] < vars_sorted[comparison_idx + 1])
            else:
                comparisons.append(vars_sorted[comparison_idx] <= vars_sorted[comparison_idx + 1])
        tri2block_mask[idx] = np.logical_and.reduce(comparisons)

    for idx, perm in enumerate(perms):
        tri2block_tp.append((0,) + tuple([p + 1 for p in perm]))

    return tri2block_map, tri2block_mask, tri2block_tp

def update_xy(mycc, t1):
    nocc, nmo = mycc.nocc, mycc.nmo
    x = np.eye(nmo, dtype=t1.dtype)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(nmo, dtype=t1.dtype)
    y[:nocc, nocc:] += t1
    return x, y

def update_fock(mycc, x, y, t1, eris):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    nocc = mycc.nocc
    t1_fock = eris.fock + einsum('risa,ia->rs', eris.pppp[:, :nocc, :, nocc:], t1) * 2.0
    t1_fock -= einsum('rias,ia->rs', eris.pppp[:, :nocc, nocc:, :], t1)
    t1_fock = x @ t1_fock @ y.T
    return t1_fock

def update_eris(mycc, x, y, eris):
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    t1_eris = einsum('tvuw,pt->pvuw', eris.pppp, x)
    t1_eris = einsum('pvuw,rv->pruw', t1_eris, x)
    t1_eris = t1_eris.transpose(2, 3, 0, 1)
    if not t1_eris.flags['C_CONTIGUOUS']:
        t1_eris = np.ascontiguousarray(t1_eris)
    t1_eris = einsum('uwpr,qu->qwpr', t1_eris, y)
    t1_eris = einsum('qwpr,sw->qspr', t1_eris, y)
    t1_eris = t1_eris.transpose(2, 3, 0, 1)
    return t1_eris

def update_t1_fock_eris(mycc, imds, t1, eris=None):
    '''Compute the Fock matrix and ERIs dressed by T1 amplitudes.'''
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    x, y = update_xy(mycc, t1)
    t1_fock = update_fock(mycc, x, y, t1, eris)
    t1_eris = update_eris(mycc, x, y, eris)
    imds.t1_fock = t1_fock
    imds.t1_eris = t1_eris
    return imds

def symmetrize_tamps_tri_(r, nocc):
    '''Symmetrize tri-stored CC amplitudes r according to equal occupied indices. E.g.,
    T3:
        - i = j <= k : symmetrize over (a, b)
        - i <= j = k : symmetrize over (b, c)
    T4:
        - i = j <= k <= l : symmetrize over (a, b)
        - i <= j = k <= l : symmetrize over (b, c)
        - i <= j <= k = l : symmetrize over (c, d)
    '''
    import numpy as np
    order = r.ndim - 1
    idx = np.meshgrid(*[np.arange(nocc)] * order, indexing='ij')
    occ = np.stack(idx, axis=-1).reshape(-1, order)
    mask = np.all(np.diff(occ, axis=1) >= 0, axis=1)
    occ_tuples = occ[mask]
    for p in range(order - 1):
        equal = np.where(occ_tuples[:, p] == occ_tuples[:, p + 1])[0]
        if equal.size == 0:
            continue
        perm = [0] + list(range(1, order + 1))
        perm[p + 1], perm[p + 2] = perm[p + 2], perm[p + 1]
        if len(equal) >= 100:
            blksize = (len(equal) + 1) // 4
            blksize = max(1, blksize)
        else:
            blksize = len(equal)
        for i in range(0, len(equal), blksize):
            i_end = min(i + blksize, len(equal))
            idx_i = equal[i:i_end]
            r_block = r[idx_i].copy()
            r_pair  = r[idx_i].transpose(perm)
            r_block += r_pair
            r_block *= 0.5
            r[idx_i] = r_block
            r_block, r_pair = None, None
    return r

def purify_tamps_tri_(r, nocc):
    '''Zero out unphysical diagonal elements in tri-stored CC amplitudes, i.e.,
    enforces T = 0 if three or more occupied/virtual indices are equal.
    '''
    import itertools, numpy as np
    order = r.ndim - 1
    # occupied indices
    idx = np.meshgrid(*[np.arange(nocc)] * order, indexing='ij')
    occ = np.stack(idx, axis=-1).reshape(-1, order)
    mask = np.all(np.diff(occ, axis=1) >= 0, axis=1)
    occ_tuples = occ[mask]
    for p in range(order - 2):
        equal = np.where((occ_tuples[:, p] == occ_tuples[:, p + 1]) & (occ_tuples[:, p + 1] == occ_tuples[:, p + 2]))[0]
        if equal.size == 0:
            continue
        r[equal, ...] = 0.0
    # virtual indices
    for perm in itertools.combinations(range(order), 3):
        idxr = [slice(None)] * order
        for p in perm:
            idxr[p] = np.mgrid[ : r.shape[p + 1]]
        r[(slice(None), ) + tuple(idxr)] = 0.0
    return r

def init_amps_rhf(mycc, eris=None):
    '''Initialize CC T-amplitudes for an RHF reference.'''
    time0 = logger.process_clock(), logger.perf_counter()
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    e_hf = mycc.e_hf
    if e_hf is None: e_hf = mycc.get_e_hf(mo_coeff=mycc.mo_coeff)

    mo_e = eris.mo_energy
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    eia = mo_e[: nocc, None] - mo_e[None, nocc :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    t1 = eris.fock[:nocc, nocc:] / eia
    t2 = eris.pppp[:nocc, :nocc, nocc:, nocc:] / eijab

    tau = t2 + einsum("ia,jb->ijab", t1, t1)
    e_corr = 2.0 * einsum("ijab,ijab->", eris.pppp[:nocc, :nocc, nocc:, nocc:], tau)
    e_corr -= einsum("ijba,ijab->", eris.pppp[:nocc, :nocc, nocc:, nocc:], tau)
    e_corr += 2.0 * einsum("ai,ia->", eris.fock[nocc:, :nocc], t1)
    logger.info(mycc, "Init t2, MP2 energy = % .12f  E_corr(MP2) % .12f" % (e_hf + e_corr, e_corr))

    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)
    cc_order = mycc.cc_order
    tamps = [t1, t2]
    for order in range(2, cc_order - 1):
        t = np.zeros((nocc,) * (order + 1) + (nvir,) * (order + 1), dtype=t1.dtype)
        tamps.append(t)
    if mycc.do_tri_max_t:
        t = np.zeros((nx(nocc, cc_order),) + (nvir,) * cc_order, dtype=t1.dtype)
    else:
        t = np.zeros((nocc,) * cc_order + (nvir,) * cc_order, dtype=t1.dtype)
    tamps.append(t)
    logger.timer(mycc, 'init mp2', *time0)
    return e_corr, tamps

def energy_rhf(mycc, tamps, eris=None):
    '''CC correlation energy for an RHF reference.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    if tamps is None:
        t1, t2 = mycc.tamps[:2]
    else:
        t1, t2 = tamps[:2]
    if eris is None: eris = mycc.ao2mo()

    nocc = t1.shape[0]

    tau = t2 + einsum("ia,jb->ijab", t1, t1)
    ed = einsum('ijab,ijab->', tau, eris.pppp[:nocc, :nocc, nocc:, nocc:]) * 2.0
    ex = - einsum('ijab,ijba->', tau, eris.pppp[:nocc, :nocc, nocc:, nocc:])

    ess = (ed * 0.5 + ex)
    ess += einsum("ai,ia->", eris.fock[nocc:, :nocc], t1) * 2.0
    eos = ed * 0.5

    if abs((ess + eos).imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in %s energy %s', mycc.__class__.__name__, ess + eos)
    mycc.e_corr = lib.tag_array((ess + eos).real, e_corr_ss=ess.real, e_corr_os=eos.real)
    return mycc.e_corr

def intermediates_t1t2(mycc, imds, t2):
    '''Intermediates for the T1 and T2 residual equation.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc = mycc.nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    F_vv = t1_fock[nocc:, nocc:].copy()
    einsum('kldc,kldb->bc', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_vv, alpha=-2.0, beta=1.0)
    einsum('klcd,kldb->bc', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_vv, alpha=1.0, beta=1.0)
    F_oo = t1_fock[:nocc, :nocc].copy()
    einsum('lkcd,ljcd->kj', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_oo, alpha=2.0, beta=1.0)
    einsum('lkdc,ljcd->kj', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_oo, alpha=-1.0, beta=1.0)
    W_oooo = t1_eris[:nocc, :nocc, :nocc, :nocc].copy()
    einsum('klcd,ijcd->klij', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_oooo, alpha=1.0, beta=1.0)
    W_ovvo = - t1_eris[:nocc, nocc:, nocc:, :nocc]
    einsum('klcd,ilad->kaci', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=-1.0, beta=1.0)
    einsum('kldc,ilad->kaci', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=0.5, beta=1.0)
    einsum('klcd,ilda->kaci', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=0.5, beta=1.0)
    W_ovov = - t1_eris[:nocc, nocc:, :nocc, nocc:]
    einsum('kldc,liad->kaic', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovov, alpha=0.5, beta=1.0)
    imds.F_vv, imds.F_oo, imds.W_oooo, imds.W_ovvo, imds.W_ovov = F_vv, F_oo, W_oooo, W_ovvo, W_ovov
    return imds

def compute_r1r2(mycc, imds, t2):
    '''Compute r1 and r2 without the contributions from T3 amplitudes.
    r2 will require a symmetry restoration step afterward.
    '''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc = mycc.nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    F_oo, F_vv, W_oooo, W_ovvo, W_ovov = imds.F_oo, imds.F_vv, imds.W_oooo, imds.W_ovvo, imds.W_ovov

    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)

    r1 = t1_fock[nocc:, :nocc].T.copy()
    einsum('kc,ikac->ia', t1_fock[:nocc, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum('akcd,ikcd->ia', t1_eris[nocc:, :nocc, nocc:, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum('klic,klac->ia', t1_eris[:nocc, :nocc, :nocc, nocc:], c_t2, out=r1, alpha=-1.0, beta=1.0)

    r2 = 0.5 * t1_eris[nocc:, nocc:, :nocc, :nocc].T
    einsum("bc,ijac->ijab", F_vv, t2, out=r2, alpha=1.0, beta=1.0)
    einsum("kj,ikab->ijab", F_oo, t2, out=r2, alpha=-1.0, beta=1.0)
    einsum("abcd,ijcd->ijab", t1_eris[nocc:, nocc:, nocc:, nocc:], t2, out=r2, alpha=0.5, beta=1.0)
    einsum("klij,klab->ijab", W_oooo, t2, out=r2, alpha=0.5, beta=1.0)
    einsum("kajc,ikcb->ijab", W_ovov, t2, out=r2, alpha=1.0, beta=1.0)
    einsum("kaci,kjcb->ijab", W_ovvo, t2, out=r2, alpha=-2.0, beta=1.0)
    einsum("kaic,kjcb->ijab", W_ovov, t2, out=r2, alpha=1.0, beta=1.0)
    einsum("kaci,jkcb->ijab", W_ovvo, t2, out=r2, alpha=1.0, beta=1.0)
    W_ovvo = imds.W_ovvo = None
    W_ovov = imds.W_ovov = None
    return r1, r2

def r1r2_add_t3_tri_(mycc, imds, r1, r2, t3):
    '''Add the T3 contributions to r1 and r2. T3 amplitudes are stored in triangular form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, nocc, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, nocc, blksize):
                bi = i1 - i0
                _unpack_t3_(mycc, t3, t3_tmp, i0, i1, j0, j1, k0, k1)
                t3_spin_summation_inplace_(t3_tmp, blksize**3, nvir, "P3_422", 1.0, 0.0)
                einsum('jkbc,ijkabc->ia', t1_eris[j0:j1, k0:k1, nocc:, nocc:],
                    t3_tmp[:bi, :bj, :bk], out=r1[i0:i1, :], alpha=0.5, beta=1.0)
    t3_tmp = None

    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, nocc, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, nocc, blksize):
                bi = i1 - i0
                _unpack_t3_(mycc, t3, t3_tmp, k0, k1, i0, i1, j0, j1)
                t3_spin_summation_inplace_(t3_tmp, blksize**3, nvir, "P3_201", 1.0, 0.0)
                einsum("kc,kijcab->ijab", t1_fock[k0:k1, nocc:], t3_tmp[:bk, :bi, :bj],
                    out=r2[i0:i1, j0:j1, :, :], alpha=0.5, beta=1.0)
                einsum("bkcd,kijdac->ijab", t1_eris[nocc:, k0:k1, nocc:, nocc:],
                        t3_tmp[:bk, :bi, :bj], out=r2[i0:i1, j0:j1, :, :], alpha=1.0, beta=1.0)
                einsum("jklc,kijcab->ilab", t1_eris[j0:j1, k0:k1, :nocc, nocc:],
                        t3_tmp[:bk, :bi, :bj], out=r2[i0:i1, :, :, :], alpha=-1.0, beta=1.0)
    t3_tmp = None
    return r1, r2

def r1r2_divide_e_(mycc, r1, r2, mo_energy):
    nocc = mycc.nocc
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:] - mycc.level_shift
    r1 /= eia
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    r2 /= eijab
    return r1, r2

def intermediates_t3(mycc, imds, t2):
    '''Intermediates for the T3 residual equation (excluding T3 contributions).'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc = mycc.nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    W_vvvv = t1_eris[nocc:, nocc:, nocc:, nocc:].copy()
    einsum('lmde,lmab->abde', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_vvvv, alpha=1.0, beta=1.0)

    W_vooo = t1_eris[nocc:, :nocc, :nocc, :nocc].copy()
    einsum('ld,ijad->alij', t1_fock[:nocc, nocc:], t2, out=W_vooo, alpha=1.0, beta=1.0)
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    einsum('mldj,mida->alij', t1_eris[:nocc, :nocc, nocc:, :nocc], c_t2, out=W_vooo, alpha=1.0, beta=1.0)
    einsum('mljd,mida->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], c_t2, out=W_vooo, alpha=-0.5, beta=1.0)
    einsum('mljd,imda->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_vooo, alpha=-0.5, beta=1.0)
    einsum('mlid,jmda->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_vooo, alpha=-1.0, beta=1.0)
    einsum('alde,ijde->alij', t1_eris[nocc:, :nocc, nocc:, nocc:], t2, out=W_vooo, alpha=1.0, beta=1.0)

    W_vvvo = t1_eris[nocc:, nocc:, nocc:, :nocc].copy()
    einsum('laed,ljeb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], c_t2, out=W_vvvo, alpha=1.0, beta=1.0)
    einsum('lade,ljeb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], c_t2, out=W_vvvo, alpha=-0.5, beta=1.0)
    einsum('lade,jleb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_vvvo, alpha=-0.5, beta=1.0)
    einsum('lbde,jlea->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_vvvo, alpha=-1.0, beta=1.0)
    einsum('lmdj,lmab->abdj', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vvvo, alpha=1.0, beta=1.0)

    W_ovvo = (2.0 * t1_eris[:nocc, nocc:, nocc:, :nocc] - t1_eris[:nocc, nocc:, :nocc, nocc:].transpose(0, 1, 3, 2))
    einsum('mled,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo, alpha=2.0, beta=1.0)
    einsum('mlde,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo, alpha=-1.0, beta=1.0)
    c_t2 = None

    W_ovov = t1_eris[:nocc, nocc:, :nocc, nocc:].copy()
    einsum('mlde,imea->laid', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovov, alpha=-1.0, beta=1.0)
    imds.W_vooo, imds.W_ovvo, imds.W_ovov, imds.W_vvvo, imds.W_vvvv = W_vooo, W_ovvo, W_ovov, W_vvvo, W_vvvv
    return imds

def intermediates_t3_add_t3_tri(mycc, imds, t3):
    '''Add the T3-dependent contributions to the T3 intermediates, with T3 stored in triangular form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize

    t1_eris = imds.t1_eris
    W_vooo, W_vvvo = imds.W_vooo, imds.W_vvvo

    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    for j0, j1 in lib.prange(0, nocc, blksize):
        bj = j1 - j0
        for l0, l1 in lib.prange(0, nocc, blksize):
            bl = l1 - l0
            for m0, m1 in lib.prange(0, nocc, blksize):
                bm = m1 - m0
                _unpack_t3_(mycc, t3, t3_tmp, m0, m1, j0, j1, l0, l1)
                t3_spin_summation_inplace_(t3_tmp, blksize**3, nvir, "P3_201", 1.0, 0.0)
                einsum('imde,mjlead->aijl', t1_eris[:nocc, m0:m1, nocc:, nocc:],
                    t3_tmp[:bm, :bj, :bl], out=W_vooo[:, :, j0:j1, l0:l1], alpha=1.0, beta=1.0)
                einsum('lmde,mjleba->abdj', t1_eris[l0:l1, m0:m1, nocc:, nocc:],
                    t3_tmp[:bm, :bj, :bl], out=W_vvvo[:, :, :, j0:j1], alpha=-1.0, beta=1.0)
    t3_tmp = None
    return imds

def compute_r3_tri(mycc, imds, t2, t3):
    '''Compute r3 with triangular-stored T3 amplitudes; r3 is returned in triangular form as well.
    r3 will require a symmetry restoration step afterward.
    '''
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize, blksize_oovv, blksize_oooo = mycc.blksize, mycc.blksize_oovv, mycc.blksize_oooo

    F_oo, F_vv = imds.F_oo, imds.F_vv
    W_oooo, W_ovvo, W_ovov, W_vvvv = imds.W_oooo, imds.W_ovvo, imds.W_ovov, imds.W_vvvv
    W_vooo, W_vvvo = imds.W_vooo, imds.W_vvvo

    r3 = np.zeros_like(t3)

    t3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize):
                bi = i1 - i0

                einsum('abdj,ikdc->ijkabc', W_vvvo[..., j0:j1], t2[i0:i1, k0:k1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=0.0)
                einsum('acdk,ijdb->ijkabc', W_vvvo[..., k0:k1], t2[i0:i1, j0:j1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum('badi,jkdc->ijkabc', W_vvvo[..., i0:i1], t2[j0:j1, k0:k1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum('bcdk,jida->ijkabc', W_vvvo[..., k0:k1], t2[j0:j1, i0:i1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum('cadi,kjdb->ijkabc', W_vvvo[..., i0:i1], t2[k0:k1, j0:j1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                einsum('cbdj,kida->ijkabc', W_vvvo[..., j0:j1], t2[k0:k1, i0:i1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)

                einsum('alij,lkbc->ijkabc', W_vooo[:, :, i0:i1, j0:j1], t2[:, k0:k1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum('alik,ljcb->ijkabc', W_vooo[:, :, i0:i1, k0:k1], t2[:, j0:j1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum('blji,lkac->ijkabc', W_vooo[:, :, j0:j1, i0:i1], t2[:, k0:k1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum('bljk,lica->ijkabc', W_vooo[:, :, j0:j1, k0:k1], t2[:, i0:i1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum('clki,ljab->ijkabc', W_vooo[:, :, k0:k1, i0:i1], t2[:, j0:j1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                einsum('clkj,liba->ijkabc', W_vooo[:, :, k0:k1, j0:j1], t2[:, i0:i1],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)

                _unpack_t3_(mycc, t3, t3_tmp, i0, i1, j0, j1, k0, k1)
                einsum('ad,ijkdbc->ijkabc', F_vv, t3_tmp[:bi, :bj, :bk], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                _unpack_t3_(mycc, t3, t3_tmp, j0, j1, i0, i1, k0, k1)
                einsum('bd,jikdac->ijkabc', F_vv, t3_tmp[:bj, :bi, :bk], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                _unpack_t3_(mycc, t3, t3_tmp, k0, k1, j0, j1, i0, i1)
                einsum('cd,kjidba->ijkabc', F_vv, t3_tmp[:bk, :bj, :bi], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)

                _accumulate_t3_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_vvvo, W_vooo, F_vv [%3d, %3d]:'%(k0, k1), *time2)
    t3_tmp = None
    r3_tmp = None
    F_vv = imds.F_vv = None
    W_vooo = imds.W_vooo = None
    W_vvvo = imds.W_vvvo = None
    time1 = log.timer_debug1('t3: W_vvvo * t2, W_vooo * t2, F_vv * t3', *time1)

    t3_tmp = np.empty((nocc,) + (blksize_oovv,) * 2 + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize_oovv,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0, k1 in lib.prange(0, nocc, blksize_oovv):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize_oovv):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize_oovv):
                bi = i1 - i0

                _unpack_t3_(mycc, t3, t3_tmp, 0, nocc, j0, j1, k0, k1, nocc, blksize_oovv, blksize_oovv)
                einsum('li,ljkabc->ijkabc', F_oo[:, i0:i1], t3_tmp[:, :bj, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=0.0)
                t3_spin_summation_inplace_(t3_tmp, nocc * blksize_oovv**2, nvir, "P3_201", 1.0, 0.0)
                einsum('ladi,ljkdbc->ijkabc', W_ovvo[..., i0:i1], t3_tmp[:, :bj, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=0.5, beta=1.0)

                _unpack_t3_(mycc, t3, t3_tmp, 0, nocc, i0, i1, k0, k1, nocc, blksize_oovv, blksize_oovv)
                einsum('lj,likbac->ijkabc', F_oo[:, j0:j1], t3_tmp[:, :bi, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                t3_spin_summation_inplace_(t3_tmp, nocc * blksize_oovv**2, nvir, "P3_201", 1.0, 0.0)
                einsum('lbdj,likdac->ijkabc', W_ovvo[..., j0:j1], t3_tmp[:, :bi, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=0.5, beta=1.0)

                _unpack_t3_(mycc, t3, t3_tmp, 0, nocc, j0, j1, i0, i1, nocc, blksize_oovv, blksize_oovv)
                einsum('lk,ljicba->ijkabc', F_oo[:, k0:k1], t3_tmp[:, :bj, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                t3_spin_summation_inplace_(t3_tmp, nocc * blksize_oovv**2, nvir, "P3_201", 1.0, 0.0)
                einsum('lcdk,ljidba->ijkabc', W_ovvo[..., k0:k1], t3_tmp[:, :bj, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=0.5, beta=1.0)

                _accumulate_t3_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1,
                                    blksize_oovv, blksize_oovv, blksize_oovv, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: F_oo, W_ovvo [%3d, %3d]:'%(k0, k1), *time2)
    t3_tmp = None
    r3_tmp = None
    F_oo = imds.F_oo = None
    W_ovvo = imds.W_ovvo = None
    time1 = log.timer_debug1('t3: F_oo * t3, W_ovvo * t3', *time1)

    t3_tmp = np.empty((blksize_oovv, nocc, blksize_oovv,) + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize_oovv,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0, k1 in lib.prange(0, nocc, blksize_oovv):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize_oovv):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize_oovv):
                bi = i1 - i0

                _unpack_t3_(mycc, t3, t3_tmp, j0, j1, 0, nocc, k0, k1, blksize_oovv, nocc, blksize_oovv)
                einsum('lbid,jlkdac->ijkabc', W_ovov[:, :, i0:i1, :], t3_tmp[:bj, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=0.0)
                _unpack_t3_(mycc, t3, t3_tmp, k0, k1, 0, nocc, j0, j1, blksize_oovv, nocc, blksize_oovv)
                einsum('lcid,kljdab->ijkabc', W_ovov[:, :, i0:i1, :], t3_tmp[:bk, :, :bj],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                _unpack_t3_pair_(mycc, t3, t3_tmp, j0, j1, 0, nocc, k0, k1)
                einsum('laid,jlkdbc->ijkabc', W_ovov[:, :, i0:i1, :], t3_tmp[:bj, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-0.5, beta=1.0)

                _unpack_t3_(mycc, t3, t3_tmp, i0, i1, 0, nocc, k0, k1, blksize_oovv, nocc, blksize_oovv)
                einsum('lajd,ilkdbc->ijkabc', W_ovov[:, :, j0:j1, :], t3_tmp[:bi, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                _unpack_t3_(mycc, t3, t3_tmp, k0, k1, 0, nocc, i0, i1, blksize_oovv, nocc, blksize_oovv)
                einsum('lcjd,klidba->ijkabc', W_ovov[:, :, j0:j1, :], t3_tmp[:bk, :, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                _unpack_t3_pair_(mycc, t3, t3_tmp, i0, i1, 0, nocc, k0, k1)
                einsum('lbjd,ilkdac->ijkabc', W_ovov[:, :, j0:j1, :], t3_tmp[:bi, :, :bk],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-0.5, beta=1.0)

                _unpack_t3_(mycc, t3, t3_tmp, i0, i1, 0, nocc, j0, j1, blksize_oovv, nocc, blksize_oovv)
                einsum('lakd,iljdcb->ijkabc', W_ovov[:, :, k0:k1, :], t3_tmp[:bi, :, :bj],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                _unpack_t3_(mycc, t3, t3_tmp, j0, j1, 0, nocc, i0, i1, blksize_oovv, nocc, blksize_oovv)
                einsum('lbkd,jlidca->ijkabc', W_ovov[:, :, k0:k1, :], t3_tmp[:bj, :, :bi],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-1.0, beta=1.0)
                _unpack_t3_pair_(mycc, t3, t3_tmp, i0, i1, 0, nocc, j0, j1)
                einsum('lckd,iljdab->ijkabc', W_ovov[:, :, k0:k1, :], t3_tmp[:bi, :, :bj],
                    out=r3_tmp[:bi, :bj, :bk], alpha=-0.5, beta=1.0)

                _accumulate_t3_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1,
                                    blksize_oovv, blksize_oovv, blksize_oovv, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_ovov [%3d, %3d]:'%(k0, k1), *time2)
    t3_tmp = None
    r3_tmp = None
    W_ovov = imds.W_ovov = None
    time1 = log.timer_debug1('t3: W_ovov * t3', *time1)

    t3_tmp = np.empty((blksize_oovv,) * 2 + (nocc,) + (nvir,) * 3, dtype=t3.dtype)
    r3_tmp = np.empty((blksize_oooo,) * 3 + (nvir,) * 3, dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for l0, l1 in lib.prange(0, nocc, blksize_oovv):
        bl = l1 - l0
        for m0, m1 in lib.prange(0, nocc, blksize_oovv):
            bm = m1 - m0
            _unpack_t3_(mycc, t3, t3_tmp, l0, l1, m0, m1, 0, nocc, blksize_oovv, blksize_oovv, nocc)
            for k0, k1 in lib.prange(0, nocc, blksize_oooo):
                bk = k1 - k0
                for j0, j1 in lib.prange(0, k1, blksize_oooo):
                    bj = j1 - j0
                    for i0, i1 in lib.prange(0, j1, blksize_oooo):
                        bi = i1 - i0
                        einsum('lmij,lmkabc->ijkabc', W_oooo[l0:l1, m0:m1, i0:i1, j0:j1],
                                t3_tmp[:bl, :bm, k0:k1], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=0.0)
                        einsum('lmik,lmjacb->ijkabc', W_oooo[l0:l1, m0:m1, i0:i1, k0:k1],
                                t3_tmp[:bl, :bm, j0:j1], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                        einsum('lmjk,lmibca->ijkabc', W_oooo[l0:l1, m0:m1, j0:j1, k0:k1],
                                t3_tmp[:bl, :bm, i0:i1], out=r3_tmp[:bi, :bj, :bk], alpha=1.0, beta=1.0)
                        _accumulate_t3_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1,
                                        blksize_oooo, blksize_oooo, blksize_oooo, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_oooo [%3d, %3d]:'%(l0, l1), *time2)
    t3_tmp = None
    r3_tmp = None
    W_oooo = imds.W_oooo = None
    time1 = log.timer_debug1('t3: W_oooo * t3', *time1)

    t3_tmp_s = np.empty((nvir, nvir, nvir), dtype=t3.dtype)
    r3_tmp_s = np.empty((nvir, nvir, nvir), dtype=t3.dtype)
    time2 = logger.process_clock(), logger.perf_counter()
    for k0 in range(nocc):
        for j0 in range(k0 + 1):
            for i0 in range(j0 + 1):
                _unpack_t3_s_pair_(mycc, t3, t3_tmp_s, i0, j0, k0)
                einsum('abde,dec->abc', W_vvvv, t3_tmp_s, out=r3_tmp_s, alpha=0.5, beta=0.0)
                _unpack_t3_s_pair_(mycc, t3, t3_tmp_s, i0, k0, j0)
                einsum('acde,deb->abc', W_vvvv, t3_tmp_s, out=r3_tmp_s, alpha=0.5, beta=1.0)
                _unpack_t3_s_pair_(mycc, t3, t3_tmp_s, j0, k0, i0)
                einsum('bcde,dea->abc', W_vvvv, t3_tmp_s, out=r3_tmp_s, alpha=0.5, beta=1.0)
                _accumulate_t3_s_(mycc, r3, r3_tmp_s, i0, j0, k0, alpha=1.0, beta=1.0)
        time2 = log.timer_debug1('t3: iter: W_vvvv %3d:'%k0, *time2)
    t3_tmp_s = None
    r3_tmp_s = None
    W_vvvv = imds.W_vvvv = None
    time1 = log.timer_debug1('t3: W_vvvv * t3', *time1)
    return r3

def r3_tri_divide_e_(mycc, r3, mo_energy):
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    blksize = mycc.blksize
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:] - mycc.level_shift
    r3_tmp = np.empty((blksize,) * 3 + (nvir,) * 3, dtype=r3.dtype)
    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, k1, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, j1, blksize):
                bi = i1 - i0
                eijkabc_blk = (eia[i0:i1, None, None, :, None, None] + eia[None, j0:j1, None, None, :, None]
                            + eia[None, None, k0:k1, None, None, :])
                _unpack_t3_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1)
                r3_tmp[:bi, :bj, :bk] /= eijkabc_blk
                _accumulate_t3_(mycc, r3, r3_tmp, i0, i1, j0, j1, k0, k1)
    eijkabc_blk = None
    r3_tmp = None
    return r3

def update_amps_rccsdt_tri_(mycc, tamps, eris):
    '''Update RCCSDT amplitudes in place, with T3 amplitudes stored in triangular form.'''
    assert (isinstance(eris, _PhysicistsERIs))

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    t1, t2, t3 = tamps
    mo_energy = eris.mo_energy

    imds = _IMDS()

    # t1 t2
    update_t1_fock_eris(mycc, imds, t1, eris)
    time1 = log.timer_debug1('update fock and eris', *time0)
    intermediates_t1t2(mycc, imds, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2(mycc, imds, t2)
    r1r2_add_t3_tri_(mycc, imds, r1, r2, t3)
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
    intermediates_t3_add_t3_tri(mycc, imds, t3)
    imds.t1_fock, imds.t1_eris = None, None
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3_tri(mycc, imds, t2, t3)
    imds = None
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # symmetrization
    symmetrize_tamps_tri_(r3, nocc)
    t3_spin_summation_inplace_(r3, r3.shape[0], nvir, "P3_full", -1.0 / 6.0, 1.0)
    purify_tamps_tri_(r3, nocc)
    time1 = log.timer_debug1('t3: symmetrize r3', *time1)
    # divide by eijkabc
    r3_tri_divide_e_(mycc, r3, mo_energy)
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm.append(np.linalg.norm(r3))

    t3 += r3
    r3 = None
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)
    return res_norm

def amplitudes_to_vector_rhf(mycc, tamps):
    '''Convert T-amplitudes to a vector form, storing only symmetry-unique elements (triangular components).'''
    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    tamps_size = [0]
    for i in range(1, len(tamps) + 1):
        tamps_size.append(nx(nocc, i) * nvir ** i)
    cum_sizes = np.cumsum(tamps_size)
    vector = np.zeros(cum_sizes[-1], dtype=tamps[0].dtype)
    for i, t in enumerate(tamps):
        idx = (*mycc.unique_tamps_map[i][0], *[slice(None)] * (i + 1))
        vector[cum_sizes[i] : cum_sizes[i + 1]] = t[idx].ravel()
    return vector

def vector_to_amplitudes_rhf(mycc, vector):
    '''Reconstruct T-amplitudes from a vector, expanding the stored unique elements into the full tensor.'''
    if mycc.unique_tamps_map is None:
        mycc.unique_tamps_map = mycc.build_unique_tamps_map()

    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    tamps_size = [0]
    for i in range(1, mycc.cc_order + 1):
        tamps_size.append(nx(nocc, i) * nvir ** i)
    cum_sizes = np.cumsum(tamps_size)

    try:
        endpoint = cum_sizes.tolist().index(vector.shape[0])
    except ValueError:
        raise ValueError("Mismatch between vector size and tamps size")
    # NOTE: Special case for two-electron systems where T3 amplitudes are empty (zero-size)
    if mycc.do_diis_max_t: endpoint = len(cum_sizes) - 1
    tamps = [None] * endpoint
    for i in range(endpoint):
        if (i == mycc.cc_order - 1) and mycc.do_tri_max_t:
            t = np.zeros((nx(nocc, i + 1),) + (nvir,) * (i + 1), dtype=vector.dtype)
        else:
            t = np.zeros((nocc,) * (i + 1) + (nvir,) * (i + 1), dtype=vector.dtype)
        idx = (*mycc.unique_tamps_map[i][0], *[slice(None)] * (i + 1))
        t[idx] = vector[cum_sizes[i] : cum_sizes[i + 1]].reshape((-1,) + (nvir,) * (i + 1))
        do_tri = mycc.do_tri_max_t and (i == mycc.cc_order - 1)
        restore_t_(t, nocc, order=i + 1, do_tri=do_tri, unique_tamps_map=mycc.unique_tamps_map[i])
        tamps[i] = t
    return tamps

def restore_t_(t, nocc, order=1, do_tri=False, unique_tamps_map=None):
    if order >= 5:
        raise NotImplementedError("restore_t function only works up to T4 amplitudes")
    if order == 2:
        if do_tri:
            raise NotImplementedError
        else:
            idx = (*unique_tamps_map[1], *[slice(None)] * order)
            t[idx] *= (1.0 / 2.0)
            t += t.transpose(1, 0, 3, 2)
    if order == 3:
        if do_tri:
            symmetrize_tamps_tri_(t, nocc)
            purify_tamps_tri_(t, nocc)
            return
        else:
            idx = (*unique_tamps_map[1], *[slice(None)] * order)
            t[idx] *= (1.0 / 2.0)
            idx = (*unique_tamps_map[2], *[slice(None)] * order)
            t[idx] *= (1.0 / 6.0)
            nocc, nvir = t.shape[0], t.shape[order]
            from pyscf.cc.rccsdt_highm import t3_perm_symmetrize_inplace_, purify_tamps_
            t3_perm_symmetrize_inplace_(t, nocc, nvir, 1.0, 0.0)
            purify_tamps_(t)
    elif order == 4:
        if do_tri:
            symmetrize_tamps_tri_(t, nocc)
            purify_tamps_tri_(t, nocc)
        else:
            idx = (*unique_tamps_map[1], *[slice(None)] * order)
            t[idx] *= (1.0 / 2.0)
            idx = (*unique_tamps_map[2], *[slice(None)] * order)
            t[idx] *= (1.0 / 6.0)
            idx = (*unique_tamps_map[3], *[slice(None)] * order)
            t[idx] *= (1.0 / 4.0)
            idx = (*unique_tamps_map[4], *[slice(None)] * order)
            t[idx] *= (1.0 / 24.0)
            nocc, nvir = t.shape[0], t.shape[order]
            from pyscf.cc.rccsdtq_highm import t4_perm_symmetrize_inplace_
            t4_perm_symmetrize_inplace_(t, nocc, nvir, 1.0, 0.0)
            from pyscf.cc.rccsdt_highm import purify_tamps_
            purify_tamps_(t)
    return t

def run_diis(mycc, tamps, istep, normt, de, adiis):
    if (adiis and istep >= mycc.diis_start_cycle and abs(de) < mycc.diis_start_energy_diff):
        vector = mycc.amplitudes_to_vector(tamps)
        tamps = mycc.vector_to_amplitudes(adiis.update(vector))
        logger.debug1(mycc, 'DIIS for step %d', istep)
    return tamps

def kernel(mycc, eris=None, tamps=None, tol=1e-8, tolnormt=1e-6, max_cycle=50, verbose=5, callback=None):
    log = logger.new_logger(mycc, verbose)

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    if tamps is None:
        tamps = mycc.init_amps(eris)[1]
    else:
        if len(tamps) < mycc.cc_order:
            tamps = list(tamps) + list(mycc.init_amps(eris)[1][len(tamps):])

    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    e_corr_old = 0.0
    e_corr = mycc.energy(tamps, eris)
    log.info('Init E_corr(%s) = %.15g', name, e_corr)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    converged = False
    mycc.cycles = 0
    for istep in range(max_cycle):
        res_norm = mycc.update_amps_(tamps, eris)

        if callback is not None:
            callback(locals())

        normt = np.linalg.norm(res_norm)

        if mycc.iterative_damping < 1.0:
            raise NotImplementedError("Damping is not implemented")

        if mycc.do_diis_max_t:
            tamps = mycc.run_diis(tamps, istep, normt, e_corr - e_corr_old, adiis)
        else:
            tamps[:mycc.cc_order - 1] = mycc.run_diis(tamps[:mycc.cc_order - 1], istep, normt,
                                                        e_corr - e_corr_old, adiis)

        e_corr_old, e_corr = e_corr, mycc.energy(tamps, eris)
        mycc.e_corr_ss = getattr(e_corr, 'e_corr_ss', 0)
        mycc.e_corr_os = getattr(e_corr, 'e_corr_os', 0)

        mycc.cycles = istep + 1
        log.info("cycle = %2d  E_corr(%s) = % .12f  dE = % .12e  norm(d tamps) = %.8e" % (
            istep + 1, mycc.__class__.__name__, e_corr, e_corr - e_corr_old, normt))
        cput1 = log.timer(f'{name} iter', *cput1)

        if abs(e_corr - e_corr_old) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    return converged, e_corr, tamps

def restore_from_diis_(mycc, diis_file, inplace=True):
    '''Reuse an existed DIIS object in the CC calculation.

    The CC amplitudes will be restored from the DIIS object. The `tamps` of the CC object will be overwritten
    by the generated `tamps`. The amplitudes vector and error vector will be reused in the CC calculation.
    '''
    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    cc_order = mycc.cc_order

    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    adiis.restore(diis_file, inplace=inplace)

    ccvec = adiis.extrapolate()
    tamps = mycc.vector_to_amplitudes(ccvec)
    if mycc.do_diis_max_t:
        mycc.tamps = tamps
    else:
        mycc.tamps[:cc_order - 1] = tamps
        if mycc.do_tri_max_t:
            mycc.tamp[-1] = np.zeros((nx(nocc, cc_order),) + (nvir,) * cc_order, dtype=ccvec.dtype)
        else:
            mycc.tamp[-1] = np.zeros((nocc,) * cc_order + (nvir,) * cc_order, dtype=ccvec.dtype)
    if inplace:
        mycc.diis = adiis
    return mycc

def _ao2mo_rcc(mycc, mo_coeff=None):
    if mycc._scf._eri is not None:
        logger.note(mycc, '_make_eris_incore_' + mycc.__class__.__name__)
        return _make_eris_incore_rcc(mycc, mo_coeff)
    elif getattr(mycc._scf, 'with_df', None):
        logger.note(mycc, '_make_df_eris_incore_' + mycc.__class__.__name__)
        return _make_df_eris_incore_rcc(mycc, mo_coeff)
    else:
        raise ValueError()

def _finalize(mycc):
    name = mycc.__class__.__name__

    if mycc.converged:
        logger.note(mycc, '%s converged', name)
    else:
        logger.note(mycc, '%s not converged', name)
    logger.note(mycc, 'E(%s) = %.16g   E_corr = %.16g', name, mycc.e_tot, mycc.e_corr)
    logger.note(mycc, 'E_corr(same-spin) = %.15g', mycc.e_corr_ss)
    logger.note(mycc, 'E_corr(oppo-spin) = %.15g', mycc.e_corr_os)
    return mycc

def build_unique_tamps_map_rhf(mycc):
    '''Build the mapping for the symmetry-unique part of the T-amplitudes.'''
    assert mycc.cc_order in (3, 4), "cc_order must be 3 or 4"
    nocc = mycc.nocc
    unique_tamps_map = []
    # t1
    unique_tamps_map.append([[slice(None)]])
    # t2
    unique_tamps_map.append([np.tril_indices(nocc), np.diag_indices(nocc)])
    # t3
    if mycc.cc_order == 3 and mycc.do_diis_max_t and mycc.do_tri_max_t:
        unique_tamps_map.append([[slice(None)]])
    elif (mycc.cc_order == 3 and mycc.do_diis_max_t and not mycc.do_tri_max_t) or mycc.cc_order == 4:
        i, j, k = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
        mask_all = (i <= j) & (j <= k)
        mask_three = (i == j) & (j == k)
        mask_two = ((i == j) | (j == k) | (i == k)) & (~mask_three) & mask_all
        unique_tamps_map.append([np.where(mask_all), np.where(mask_two), np.where(mask_three)])
    # t4
    if mycc.cc_order == 4 and mycc.do_diis_max_t:
        if mycc.do_tri_max_t:
            unique_tamps_map.append([[slice(None)]])
        else:
            i, j, k, l = np.meshgrid(np.arange(nocc), np.arange(nocc), np.arange(nocc), np.arange(nocc), indexing='ij')
            mask_all = (i <= j) & (j <= k) & (k <= l)
            mask_four = (i == j) & (j == k) & (k == l)
            mask_three = (((i == j) & (j == k) & (k < l)) | ((i < j) & (j == k) & (k == l))) & mask_all
            mask_three_2 = ((i == j) & (j < k) & (k == l)) & mask_all
            mask_two = (((i == j) & (j < k) & (k < l)) | ((i < j) & (j == k) & (k < l))
                        | ((i < j) & (j < k) & (k == l))) & mask_all
            unique_tamps_map.append([np.where(mask_all), np.where(mask_two), np.where(mask_three),
                                    np.where(mask_three_2), np.where(mask_four)])
    return unique_tamps_map

def dump_flags(mycc, verbose=None):
    log = logger.new_logger(mycc, verbose)
    log.info('')
    log.info('******** %s ********', mycc.__class__)
    log.info('%s nocc = %s, nmo = %s', mycc.__class__.__name__, mycc.nocc, mycc.nmo)
    if mycc.do_tri_max_t:
        text = '<='.join('ijklml'[:mycc.cc_order])
        log.info("Allocating only the %s part of the T%d amplitude in memory", text, mycc.cc_order)
    else:
        log.info("Allocating the entire T%d amplitude in memory", mycc.cc_order)
    if mycc.frozen is not None:
        log.info('frozen orbitals %s', mycc.frozen)
    log.info('max_cycle = %d', mycc.max_cycle)
    log.info('conv_tol = %g', mycc.conv_tol)
    log.info('conv_tol_normt = %s', mycc.conv_tol_normt)
    if mycc.do_diis_max_t:
        log.info('diis with the T%d amplitude', mycc.cc_order)
    else:
        log.info('diis without the T%d amplitude', mycc.cc_order)
    log.info('diis_space = %d', mycc.diis_space)
    if mycc.diis_file:
        log.info('diis_file = %s', mycc.diis_file)
    log.info('diis_start_cycle = %d', mycc.diis_start_cycle)
    log.info('diis_start_energy_diff = %g', mycc.diis_start_energy_diff)
    log.info('max_memory %d MB (current use %d MB)', mycc.max_memory, lib.current_memory()[0])
    if mycc.einsum_backend is not None:
        log.info('einsum_backend: %s', mycc.einsum_backend)
    return mycc

def vector_size_rhf(mycc, nmo=None, nocc=None):
    from math import prod, factorial
    nx = lambda n, order: prod(n + i for i in range(order)) // factorial(order)

    if nocc is None: nocc = mycc.nocc
    if nmo is None: nmo = mycc.nmo
    nvir = nmo - nocc
    tamps_size = [0]
    # TODO: Should this function take `do_diis_max_t` into account?
    for i in range(1, mycc.cc_order + 1):
        tamps_size.append(nx(nocc, i) * nvir ** i)
    cum_sizes = np.cumsum(tamps_size)
    return cum_sizes[-1]

def format_size(i, suffix='B'):
    if i < 1000:
        return "%d %s" % (i, suffix)
    else:
        a = 1024
        for pf in "KMGTPEZY":
            p = 2
            for k in [10, 100, 1000]:
                if i < k * a:
                    return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                p -= 1
            a *= 1024
    return "??? " + suffix

def memory_estimate_log_rccsdt(mycc):
    '''Estimate the memory cost.'''
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc

    log.info('Approximate memory usage estimate')
    if mycc.do_tri_max_t:
        nocc3 = nocc * (nocc + 1) * (nocc + 2) // 6
        t3_memory = nocc3 * nvir**3 * 8
    else:
        t3_memory = nocc**3 * nvir**3 * 8
    log.info('    T3 memory               %8s', format_size(t3_memory))
    log.info('    R3 memory               %8s', format_size(t3_memory))

    if not mycc.do_tri_max_t:
        symm_t3_memory = t3_memory
        log.info('    Symmetrized T3 memory   %8s', format_size(symm_t3_memory))
    if mycc.do_tri_max_t:
        if nocc * (nocc + 1) // 2 >= 100:
            factor = 4
        else:
            factor = 1
        symm_t3_memory = nocc * (nocc + 1) // 2 * nvir**3 * 8 * 2 / factor
        log.info('    Symmetrized T3 memory   %8s', format_size(symm_t3_memory))

    eris_memory = nmo**4 * 8
    log.info('    ERIs memory             %8s', format_size(eris_memory))
    log.info('    T1-ERIs memory          %8s', format_size(eris_memory))
    log.info('    Intermediates memory    %8s', format_size(eris_memory))

    if mycc.do_tri_max_t:
        blk_memory = (mycc.blksize_oovv**2 * nocc * nvir**3 + mycc.blksize_oooo**3 * nvir**3) * 8
        log.info("    Block workspace         %8s", format_size(blk_memory))

    if mycc.einsum_backend in ['numpy', 'pyscf']:
        if mycc.do_tri_max_t:
            einsum_memory = blk_memory
            log.info("    T3 einsum buffer        %8s", format_size(einsum_memory))
        else:
            einsum_memory = t3_memory
            log.info("    T3 einsum buffer        %8s", format_size(einsum_memory))

    if mycc.incore_complete:
        if mycc.do_diis_max_t:
            diis_memory = nocc * (nocc + 1) * (nocc + 2) // 6 * nvir**3 * 8 * mycc.diis_space * 2
        else:
            diis_memory = nocc * (nocc + 1) // 2 * nvir**2 * 8 * mycc.diis_space * 2
        log.info('    DIIS memory             %8s', format_size(diis_memory))
    else:
        diis_memory = 0.0

    total_memory = 2 * t3_memory + symm_t3_memory + 3 * eris_memory + diis_memory
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
            logger.warn(mycc, 'Consider using %s in `pyscf.cc.rccsdt` which stores the triangular T amplitudes',
                        mycc.__class__.__name__)
        else:
            logger.warn(mycc, 'Consider reducing `blksize`, `blksize_oooo`, and `blksize_oovv` to reduce memory usage')
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
        lib.chkfile.save(mycc.chkfile, 'rccsdt', cc_chk)
    else:
        lib.chkfile.save(mycc.chkfile, 'rccsdt_highm', cc_chk)
    return mycc

def tamps_tri2full_rhf(mycc, tamps_tri):
    '''Convert triangular-stored T amplitudes to their full tensor form.'''
    assert mycc.cc_order in (3, 4), "cc_order must be 3 or 4"
    if mycc.cc_order == 3:
        assert tamps_tri.ndim == 4, "tamps_tri.ndim must be 4 for t3_tri"
    elif mycc.cc_order == 4:
        assert tamps_tri.ndim == 5, "tamps_tri.ndim must be 5 for t4_tri"
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    tamps_full = np.zeros((nocc,) * mycc.cc_order + (nvir,) * mycc.cc_order, dtype=tamps_tri.dtype)
    if mycc.cc_order == 3:
        _unpack_t3_(mycc, tamps_tri, tamps_full, 0, nocc, 0, nocc, 0, nocc, blksize0=nocc, blksize1=nocc, blksize2=nocc)
    elif mycc.cc_order == 4:
        from pyscf.cc.rccsdtq import _unpack_t4_
        _unpack_t4_(mycc, tamps_tri, tamps_full, 0, nocc, 0, nocc, 0, nocc, 0, nocc,
                        blksize0=nocc, blksize1=nocc, blksize2=nocc, blksize3=nocc)
    return tamps_full

def tamps_full2tri_rhf(mycc, tamps_full):
    '''Convert full T amplitudes to their triangular-stored form.'''
    nocc, order = tamps_full.shape[0], tamps_full.ndim // 2
    idx = np.meshgrid(*[np.arange(nocc)] * order, indexing='ij')
    occ = np.stack(idx, axis=-1)
    mask = np.all(np.diff(occ, axis=-1) >= 0, axis=-1)
    occ_tuple_idx = np.where(mask)
    full_index = tuple(occ_tuple_idx) + (slice(None),) * order
    tamps_tri = tamps_full[full_index]
    return tamps_tri


class RCCSDT(ccsd.CCSDBase):

    conv_tol = getattr(__config__, 'cc_rccsdt_RCCSDT_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_rccsdt_RCCSDT_conv_tol_normt', 1e-6)
    cc_order = getattr(__config__, 'cc_rccsdt_RCCSDT_cc_order', 3)
    do_diis_max_t = getattr(__config__, 'cc_rccsdt_RCCSDT_do_diis_max_t', True)
    blksize = getattr(__config__, 'cc_rccsdt_RCCSDT_blksize', 8)
    blksize_oovv = getattr(__config__, 'cc_rccsdt_RCCSDT_blksize_oovv', 4)
    blksize_oooo = getattr(__config__, 'cc_rccsdt_RCCSDT_blksize_oooo', 4)
    einsum_backend = getattr(__config__, 'cc_rccsdt_RCCSDT_einsum_backend', 'numpy')

    _keys = {
        'max_cycle', 'conv_tol', 'iterative_damping', 'conv_tol_normt', 'diis', 'diis_space', 'diis_file',
        'diis_start_cycle', 'diis_start_energy_diff', 'async_io', 'incore_complete', 'callback',
        'mol', 'verbose', 'stdout', 'frozen', 'level_shift', 'mo_coeff', 'mo_occ', 'cycles', 'emp2', 'e_hf',
        'converged', 'e_corr', 'chkfile', 'cc_order', 'do_diis_max_t', 'blksize', 'blksize_oovv', 'blksize_oooo',
        'einsum_backend', 'tamps', 'unique_tamps_map', 'tri2block_map', 'tri2block_mask', 'tri2block_tp',
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        self.tamps = [None, None, None]
        ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.unique_tamps_map = None

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

    do_tri_max_t = property(lambda self: True)

    def set_einsum_backend(self, backend):
        self.einsum_backend = backend

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf
    ao2mo = _ao2mo_rcc
    init_amps = init_amps_rhf
    energy = energy_rhf
    restore_from_diis_ = restore_from_diis_
    memory_estimate_log = memory_estimate_log_rccsdt
    update_amps_ = update_amps_rccsdt_tri_
    amplitudes_to_vector = amplitudes_to_vector_rhf
    vector_to_amplitudes = vector_to_amplitudes_rhf
    build_unique_tamps_map = build_unique_tamps_map_rhf
    setup_tri2block = setup_tri2block_rhf
    vector_size = vector_size_rhf
    run_diis = run_diis
    _finalize = _finalize
    dump_flags = dump_flags
    dump_chk = dump_chk
    tamps_tri2full = tamps_tri2full_rhf
    tamps_full2tri = tamps_full2tri_rhf

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

        nocc = eris.nocc

        if self.do_tri_max_t:
            self.tri2block_map, self.tri2block_mask, self.tri2block_tp = self.setup_tri2block()

            self.blksize = min(self.blksize, nocc)
            self.blksize_oovv = min(self.blksize_oovv, nocc)
            self.blksize_oooo = min(self.blksize_oooo, nocc)
            log.info('blksize %2d    blksize_oovv %2d    blksize_oooo %2d    blksize_vvvv %2d'%(
                        self.blksize, self.blksize_oovv, self.blksize_oooo, 1))
            if self.blksize > (nocc + 1) // 2:
                logger.warn(self, 'A large `blksize` is being used, which may cause large memory consumption\n'
                            '      for storing contraction intermediates. If memory is sufficient, consider using\n'
                            '      `pyscf.cc.rccsdt_highm.RCCSDT` instead.')
            if self.blksize_oovv > (nocc + 1) // 2:
                logger.warn(self, 'A large `blksize_oovv` is being used, which may cause large memory consumption\n'
                            '      for storing contraction intermediates. If memory is sufficient, consider using\n'
                            '      `pyscf.cc.rccsdt_highm.RCCSDT` instead.')
            if self.blksize_oooo > (nocc + 1) // 2:
                logger.warn(self, 'A large `blksize_oooo` is being used, which may cause large memory consumption\n'
                            '      for storing contraction intermediates. If memory is sufficient, consider using\n'
                            '      `pyscf.cc.rccsdt_highm.RCCSDT` instead.')

        self.memory_estimate_log()
        self.unique_tamps_map = self.build_unique_tamps_map()

        self.converged, self.e_corr, self.tamps = kernel(self, eris, tamps, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt, verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.tamps

    def ccsdt_q(self, tamps, eris=None):
        raise NotImplementedError

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

class _PhysicistsERIs:
    '''<pq|rs> = (pr|qs)'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(numpy.dot, (mo_coeff.conj().T, fockao, mo_coeff))

        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.mo_energy = self.fock.diagonal().real
        try:
            gap = abs(mo_e[:nocc,None] - mo_e[None,nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for %s.\n %s may be difficult to converge.'
                            'Increasing %s Attribute level_shift may improve convergence.',
                            gap, mycc.__class__.__name__, mycc.__class__.__name__, mycc.__class__.__name__)
        except ValueError:  # gap.size == 0
            pass
        return self

def _make_eris_incore_rcc(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nmo = eris.fock.shape[0]

    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    eris.pppp = ao2mo.restore(1, eri1, nmo).transpose(0, 2, 1, 3)
    if not eris.pppp.flags['C_CONTIGUOUS']:
        eris.pppp = np.ascontiguousarray(eris.pppp)

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris

def _make_df_eris_incore_rcc(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mo_coeff = numpy.asarray(eris.mo_coeff, order='F')
    nao, nmo = mo_coeff.shape

    naux = mycc._scf.with_df.get_naoaux()
    ijslice = (0, nmo, 0, nmo)
    Lpq = numpy.empty((naux, nmo, nmo))
    p1 = 0
    Lpq_tmp = None
    for eri1 in mycc._scf.with_df.loop():
        Lpq_tmp = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq_tmp).reshape(-1, nmo, nmo)
        p0, p1 = p1, p1 + Lpq_tmp.shape[0]
        Lpq[p0:p1, :, :] = Lpq_tmp[:, :, :]
        Lpq_tmp = None
    Lpq = Lpq.reshape(naux, nmo * nmo)

    eris.pppp = lib.ddot(Lpq.T, Lpq).reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
    if not eris.pppp.flags['C_CONTIGUOUS']:
        eris.pppp = np.ascontiguousarray(eris.pppp)

    logger.timer(mycc, mycc.__class__.__name__ + ' integral transformation', *cput0)
    return eris


if __name__ == "__main__":

    from pyscf import gto, scf
    from pyscf.data.elements import chemcore

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", verbose=3)
    mf = scf.RHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()
    print()

    ref_e_corr = -0.3217858674891447
    mycc = RCCSDT(mf, frozen=chemcore(mol))
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.do_diis_max_t = True
    mycc.incore_complete = True
    mycc.blksize = 3
    mycc.blksize_oovv = 2
    mycc.blksize_oooo = 3
    mycc.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_e_corr, mycc.e_corr - ref_e_corr))
    print('\n' * 5)

    # comparison with the high-memory version
    from pyscf.cc.rccsdt_highm import RCCSDT as RCCSDThm
    mycc2 = RCCSDThm(mf, frozen=chemcore(mol))
    mycc2.set_einsum_backend('numpy')
    mycc2.conv_tol = 1e-12
    mycc2.conv_tol_normt = 1e-10
    mycc2.max_cycle = 100
    mycc2.verbose = 5
    mycc2.do_diis_max_t = True
    mycc2.incore_complete = True
    mycc2.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc2.e_corr, ref_e_corr, mycc2.e_corr - ref_e_corr))
    print()

    t3_tri = mycc.t3
    t3_full = mycc2.t3
    t3_tri_from_t3_full = mycc2.tamps_full2tri(t3_full)
    t3_full_from_t3_tri = mycc.tamps_tri2full(t3_tri)

    print('energy difference                          % .10e' % (mycc.e_tot - mycc2.e_tot))
    print('max(abs(t1 difference))                    % .10e' % np.max(np.abs(mycc.t1 - mycc2.t1)))
    print('max(abs(t2 difference))                    % .10e' % np.max(np.abs(mycc.t2 - mycc2.t2)))
    print('max(abs(t3_tri - t3_tri_from_t3_full))     % .10e' % np.max(np.abs(t3_tri - t3_tri_from_t3_full)))
    print('max(abs(t3_full - t3_full_from_t3_tri))    % .10e' % np.max(np.abs(t3_full - t3_full_from_t3_tri)))
