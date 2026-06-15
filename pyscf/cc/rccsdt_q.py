#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
RHF-CCSDT(Q) for real integrals
'''

import functools
import numpy as np
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.rccsdt import _einsum, _unpack_t3_, setup_tri2block_rhf
from pyscf.cc.rccsdtq import t4_add_


_libccsdt = lib.load_library('libccsdt')

def eijkl_division_single_(A, eocc, evir, i, j, k, l, nvir):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    assert eocc.dtype == np.float64 and eocc.flags['C_CONTIGUOUS'], "eocc must be a contiguous float64 array"
    assert evir.dtype == np.float64 and evir.flags['C_CONTIGUOUS'], "evir must be a contiguous float64 array"
    _libccsdt.eijkl_division_single_(
        A.ctypes.data_as(ctypes.c_void_p), eocc.ctypes.data_as(ctypes.c_void_p), evir.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(i), ctypes.c_int64(j), ctypes.c_int64(k), ctypes.c_int64(l), ctypes.c_int64(nvir)
    )
    return A

def t4_spin_summation_single_inplace_(A, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == np.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"
    pattern_c = pattern.encode('utf-8')
    _libccsdt.t4_spin_summation_single_inplace_(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nvir), ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha), ctypes.c_double(beta)
    )
    return A

def kernel(mycc, eris=None, tamps=None, verbose=logger.NOTE):

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mycc, verbose)

    if tamps is not None:
        if len(tamps) != 3:
            raise ValueError("tamps should be a list of length 3, containing T1, T2, and T3 amplitudes.")
        if mycc.do_tri_max_t and len(tamps[2].shape) != 4:
            raise ValueError("CC object uses compact T3 amplitudes but the input T3 is full.")
        if not mycc.do_tri_max_t and len(tamps[2].shape) == 4:
            raise ValueError("CC object uses full T3 amplitudes but the input T3 is compact.")
    else:
        tamps = mycc.tamps

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)

    if mycc.do_tri_max_t and (not hasattr(mycc, "tri2block_map") or mycc.tri2block_map is None):
        mycc.tri2block_map, mycc.tri2block_mask, mycc.tri2block_tp = setup_tri2block_rhf(mycc)

    name = mycc.__class__.__name__

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)

    t1 = tamps[0]
    nocc, nvir = t1.shape[0], t1.shape[1]

    t2, t3 = tamps[1:3]
    mo_energy = eris.mo_energy
    e_occ = mo_energy[:nocc]
    e_occ = np.ascontiguousarray(e_occ)
    e_vir = mo_energy[nocc:]
    e_vir = np.ascontiguousarray(e_vir)

    eris_ovvv = eris.pppp[:nocc, nocc:, nocc:, nocc:].copy()
    eris_oovo = eris.pppp[:nocc, :nocc, nocc:, :nocc].copy()
    eris_oovv = eris.pppp[:nocc, :nocc, nocc:, nocc:].copy()
    eris_ovvo = eris.pppp[:nocc, nocc:, nocc:, :nocc].copy()
    eris_ovov = eris.pppp[:nocc, nocc:, :nocc, nocc:].copy()
    eris_vvvv = eris.pppp[nocc:, nocc:, nocc:, nocc:].copy()
    eris_oooo = eris.pppp[:nocc, :nocc, :nocc, :nocc].copy()

    eris = None

    def get_t3_slice(t3_blk, i, j):
        if mycc.do_tri_max_t:
            _unpack_t3_(mycc, t3, t3_blk, i, i + 1, j, j + 1, 0, nocc, 1, 1, nocc)
        else:
            t3_blk[0, 0, :nocc] = t3[i, j, :nocc]
        return t3_blk

    def compute_W_vvvvoo(W_vvvvoo_slice, j, k):
        einsum('abef,fc->abce', eris_vvvv, t2[j, k], out=W_vvvvoo_slice, alpha=0.5, beta=0.0)
        einsum('acef,fb->abce', eris_vvvv, t2[k, j], out=W_vvvvoo_slice, alpha=0.5, beta=1.0)
        return W_vvvvoo_slice

    def compute_W_vvoooo(W_vvoooo_slice, i, j, k):
        einsum('eam,be->abm', eris_ovvo[i], t2[j, k], out=W_vvoooo_slice, alpha=1.0, beta=0.0)
        einsum('ebm,ae->abm', eris_ovvo[j], t2[i, k], out=W_vvoooo_slice, alpha=1.0, beta=1.0)
        einsum('ema,be->abm', eris_ovov[k], t2[j, i], out=W_vvoooo_slice, alpha=1.0, beta=1.0)
        einsum('emb,ae->abm', eris_ovov[k], t2[i, j], out=W_vvoooo_slice, alpha=1.0, beta=1.0)
        einsum('mn,nab->abm', eris_oooo[k, i], t2[:, j], out=W_vvoooo_slice, alpha=-0.5, beta=1.0)
        einsum('mn,nba->abm', eris_oooo[k, j], t2[:, i], out=W_vvoooo_slice, alpha=-0.5, beta=1.0)
        return W_vvoooo_slice

    time1 = logger.process_clock(), logger.perf_counter()
    t4_blk = np.empty((nvir,) * 4, dtype=t2.dtype)
    z4_blk = np.empty_like(t4_blk)
    t3_blk = np.empty((1,) * 2 + (nocc,) + (nvir,) * 3, dtype=t3.dtype)
    W_vvoooo_slice = np.empty((nvir, nvir, nocc), dtype=t2.dtype)
    W_vvvvoo_slice = np.empty((nvir, nvir, nvir, nvir), dtype=t2.dtype)
    e_q_bracket = 0.0
    e_q_paren = 0.0
    for l in range(nocc):
        for k in range(l + 1):
            for j in range(k + 1):
                for i in range(j + 1):

                    if (i == j == k == l) or (i == j and j == k) or (j == k and k == l):
                        continue
                    elif i < j and j < k and k < l:
                        factor = 24.0
                    elif (i == j and j < k and k < l) or (i < j and j == k and k < l) or (i < j and j < k and k == l):
                        factor = 12.0
                    elif (i == j and j < k and k == l):
                        factor = 6.0

                    # z for (Q)
                    get_t3_slice(t3_blk, k, l)
                    einsum('am,mcdb->abcd', eris_oovo[i, j], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=0.0)
                    einsum('bm,mcda->abcd', eris_oovo[j, i], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('eba,cde->abcd', eris_ovvv[j], t3_blk[0, 0, i], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('eab,cde->abcd', eris_ovvv[i], t3_blk[0, 0, j], out=z4_blk, alpha=1.0, beta=1.0)

                    get_t3_slice(t3_blk, j, l)
                    einsum('am,mbdc->abcd', eris_oovo[i, k], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('cm,mbda->abcd', eris_oovo[k, i], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('eca,bde->abcd', eris_ovvv[k], t3_blk[0, 0, i], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('eac,bde->abcd', eris_ovvv[i], t3_blk[0, 0, k], out=z4_blk, alpha=1.0, beta=1.0)

                    get_t3_slice(t3_blk, j, k)
                    einsum('am,mbcd->abcd', eris_oovo[i, l], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('dm,mbca->abcd', eris_oovo[l, i], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('eda,bce->abcd', eris_ovvv[l], t3_blk[0, 0, i], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('ead,bce->abcd', eris_ovvv[i], t3_blk[0, 0, l], out=z4_blk, alpha=1.0, beta=1.0)

                    get_t3_slice(t3_blk, i, l)
                    einsum('bm,madc->abcd', eris_oovo[j, k], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('cm,madb->abcd', eris_oovo[k, j], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('ecb,ade->abcd', eris_ovvv[k], t3_blk[0, 0, j], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('ebc,ade->abcd', eris_ovvv[j], t3_blk[0, 0, k], out=z4_blk, alpha=1.0, beta=1.0)

                    get_t3_slice(t3_blk, i, k)
                    einsum('bm,macd->abcd', eris_oovo[j, l], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('dm,macb->abcd', eris_oovo[l, j], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('edb,ace->abcd', eris_ovvv[l], t3_blk[0, 0, j], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('ebd,ace->abcd', eris_ovvv[j], t3_blk[0, 0, l], out=z4_blk, alpha=1.0, beta=1.0)

                    get_t3_slice(t3_blk, i, j)
                    einsum('cm,mabd->abcd', eris_oovo[k, l], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('dm,mabc->abcd', eris_oovo[l, k], t3_blk[0, 0], out=z4_blk, alpha=-1.0, beta=1.0)
                    einsum('edc,abe->abcd', eris_ovvv[l], t3_blk[0, 0, k], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('ecd,abe->abcd', eris_ovvv[k], t3_blk[0, 0, l], out=z4_blk, alpha=1.0, beta=1.0)

                    # t4
                    compute_W_vvoooo(W_vvoooo_slice, i, j, k)
                    einsum('abm,mdc->abcd', W_vvoooo_slice, t2[l], out=t4_blk, alpha=-1.0, beta=0.0)
                    compute_W_vvoooo(W_vvoooo_slice, i, j, l)
                    einsum('abm,mcd->abcd', W_vvoooo_slice, t2[k], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, i, k, j)
                    einsum('acm,mdb->abcd', W_vvoooo_slice, t2[l], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, i, k, l)
                    einsum('acm,mbd->abcd', W_vvoooo_slice, t2[j], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, i, l, j)
                    einsum('adm,mcb->abcd', W_vvoooo_slice, t2[k], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, i, l, k)
                    einsum('adm,mbc->abcd', W_vvoooo_slice, t2[j], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, j, k, i)
                    einsum('bcm,mda->abcd', W_vvoooo_slice, t2[l], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, j, k, l)
                    einsum('bcm,mad->abcd', W_vvoooo_slice, t2[i], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, j, l, i)
                    einsum('bdm,mca->abcd', W_vvoooo_slice, t2[k], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, j, l, k)
                    einsum('bdm,mac->abcd', W_vvoooo_slice, t2[i], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, k, l, i)
                    einsum('cdm,mba->abcd', W_vvoooo_slice, t2[j], out=t4_blk, alpha=-1.0, beta=1.0)
                    compute_W_vvoooo(W_vvoooo_slice, k, l, j)
                    einsum('cdm,mab->abcd', W_vvoooo_slice, t2[i], out=t4_blk, alpha=-1.0, beta=1.0)

                    compute_W_vvvvoo(W_vvvvoo_slice, j, k)
                    einsum('abce,ed->abcd', W_vvvvoo_slice, t2[i, l], out=t4_blk, alpha=1.0, beta=1.0)
                    einsum('dbce,ea->abcd', W_vvvvoo_slice, t2[l, i], out=t4_blk, alpha=1.0, beta=1.0)
                    compute_W_vvvvoo(W_vvvvoo_slice, j, l)
                    einsum('abde,ec->abcd', W_vvvvoo_slice, t2[i, k], out=t4_blk, alpha=1.0, beta=1.0)
                    einsum('cbde,ea->abcd', W_vvvvoo_slice, t2[k, i], out=t4_blk, alpha=1.0, beta=1.0)
                    compute_W_vvvvoo(W_vvvvoo_slice, k, l)
                    einsum('acde,eb->abcd', W_vvvvoo_slice, t2[i, j], out=t4_blk, alpha=1.0, beta=1.0)
                    einsum('bcde,ea->abcd', W_vvvvoo_slice, t2[j, i], out=t4_blk, alpha=1.0, beta=1.0)
                    compute_W_vvvvoo(W_vvvvoo_slice, i, k)
                    einsum('bace,ed->abcd', W_vvvvoo_slice, t2[j, l], out=t4_blk, alpha=1.0, beta=1.0)
                    einsum('dace,eb->abcd', W_vvvvoo_slice, t2[l, j], out=t4_blk, alpha=1.0, beta=1.0)
                    compute_W_vvvvoo(W_vvvvoo_slice, i, l)
                    einsum('bade,ec->abcd', W_vvvvoo_slice, t2[j, k], out=t4_blk, alpha=1.0, beta=1.0)
                    einsum('cade,eb->abcd', W_vvvvoo_slice, t2[k, j], out=t4_blk, alpha=1.0, beta=1.0)
                    compute_W_vvvvoo(W_vvvvoo_slice, i, j)
                    einsum('cabe,ed->abcd', W_vvvvoo_slice, t2[k, l], out=t4_blk, alpha=1.0, beta=1.0)
                    einsum('dabe,ec->abcd', W_vvvvoo_slice, t2[l, k], out=t4_blk, alpha=1.0, beta=1.0)

                    t4_add_(t4_blk, z4_blk, 1, nvir)
                    eijkl_division_single_(t4_blk, e_occ, e_vir, i, j, k, l, nvir)
                    t4_spin_summation_single_inplace_(t4_blk, nvir, 'P4_444', alpha=1.0, beta=0.0)

                    e_q_paren += np.dot(z4_blk.ravel(), t4_blk.ravel()) * factor

                    # z for [Q]
                    einsum('ab,cd->abcd', eris_oovv[i, j], t2[k, l], out=z4_blk, alpha=1.0, beta=0.0)
                    einsum('ac,bd->abcd', eris_oovv[i, k], t2[j, l], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('ad,bc->abcd', eris_oovv[i, l], t2[j, k], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('bc,ad->abcd', eris_oovv[j, k], t2[i, l], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('bd,ac->abcd', eris_oovv[j, l], t2[i, k], out=z4_blk, alpha=1.0, beta=1.0)
                    einsum('cd,ab->abcd', eris_oovv[k, l], t2[i, j], out=z4_blk, alpha=1.0, beta=1.0)

                    e_q_bracket += np.dot(z4_blk.ravel(), t4_blk.ravel()) * factor

        time1 = log.timer_debug1('%s(Q): iter %3d:' % (name, l), *time1)

    e_q_paren += e_q_bracket
    e_q_bracket /= 12.0
    e_q_paren /= 12.0

    log.timer('%s(Q)' % name, *time0)
    log.info("[Q] correction = % .12e    (Q) correction = % .12e" % (e_q_bracket, e_q_paren))
    return e_q_bracket, e_q_paren


if __name__ == '__main__':

    from pyscf import gto, scf, lib
    from pyscf.data.elements import chemcore
    from pyscf.cc.rccsdt import RCCSDT
    from pyscf.cc.rccsdt_highm import RCCSDT as RCCSDT_highm

    atom = '''
    O  1.416468653903   0.111264435953   0.000000000000
    H  1.746241653903  -0.373945564047  -0.758561000000
    H  2.102765241     -0.898304829      1.578786622
    '''
    basis = 'cc-pvdz'

    mol = gto.M(atom=atom, basis=basis)
    mol.verbose = 1
    mol.max_memory = 10000
    frozen = chemcore(mol)

    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()

    mycc = RCCSDT(mf, frozen=frozen)
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 100
    mycc.verbose = 3
    mycc.blksize = 2
    mycc.blksize_oovv = 2
    mycc.blksize_oooo = 2
    mycc.do_diis_max_t = False
    mycc.incore_complete = True
    ecorr, tamps = mycc.kernel()

    ref_e_q_bracket = -0.001462052703
    ref_e_q_paren = -0.001620887567

    mycc.verbose = 8
    e_q_bracket, e_q_paren = kernel(mycc)
    print('[Q] corr: % .12f    Ref: % .12f    Diff: % .12e'%(
        e_q_bracket, ref_e_q_bracket, e_q_bracket - ref_e_q_bracket))
    print('(Q) corr: % .12f    Ref: % .12f    Diff: % .12e'%(
        e_q_paren, ref_e_q_paren, e_q_paren - ref_e_q_paren))

    mycc2 = RCCSDT_highm(mf, frozen=frozen)
    mycc2.set_einsum_backend('numpy')
    mycc2.conv_tol = 1e-10
    mycc2.conv_tol_normt = 1e-8
    mycc2.max_cycle = 100
    mycc2.verbose = 3
    mycc2.do_diis_max_t = False
    mycc2.incore_complete = True
    ecorr, tamps = mycc2.kernel()
    e_q_bracket, e_q_paren = mycc2.ccsdt_q()
    print('[Q] corr: % .12f    Ref: % .12f    Diff: % .12e'%(
        e_q_bracket, ref_e_q_bracket, e_q_bracket - ref_e_q_bracket))
    print('(Q) corr: % .12f    Ref: % .12f    Diff: % .12e'%(
        e_q_paren, ref_e_q_paren, e_q_paren - ref_e_q_paren))
