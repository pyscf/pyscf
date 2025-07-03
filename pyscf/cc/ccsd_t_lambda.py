#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Yu Jin <jinyuchem@uchicago.edu>
#

'''
Spin-free lambda equation of RHF-CCSD(T)

Ref:
JCP 147, 044104 (2017); DOI:10.1063/1.4994918
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd, ccsd_lambda, _ccsd

# Note: not support fov != 0

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)

    if numpy.iscomplexobj(t1) or numpy.iscomplexobj(t2) or numpy.iscomplexobj(eris):
        raise ValueError("make_intermediates does not support complex-valued inputs (t1, t2, or eris)")

    imds = ccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovov)

    mo_e = eris.mo_energy
    eia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])

    imds.l1_t = numpy.zeros((nocc, nvir), dtype=t1.dtype)
    joovv = numpy.zeros((nocc, nocc, nvir, nvir), dtype=t1.dtype)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nvir, int(((max_memory * 0.9e6/8) / 6.0 / (nocc**3))**(1/3)))
    if blksize < nvir:
        blksize = min(blksize, (nvir + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1('ccsd_t lambda make_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    w_blk = numpy.empty((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)
    v_blk = numpy.empty((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)

    time2 = logger.process_clock(), logger.perf_counter()
    for c0, c1 in lib.prange(0, nvir, blksize):
        bc = c1 - c0
        for b0, b1 in lib.prange(0, nvir, blksize):
            bb = b1 - b0
            for a0, a1 in lib.prange(0, nvir, blksize):
                ba = a1 - a0

                w_blk[:ba, :bb, :bc] = lib.einsum('iafb,kjcf->abcijk',
                    eris_ovvv[:, a0:a1, :, b0:b1], t2[:, :, c0:c1, :])
                w_blk[:ba, :bb, :bc] += lib.einsum('iafc,jkbf->abcijk',
                    eris_ovvv[:, a0:a1, :, c0:c1], t2[:, :, b0:b1, :])
                w_blk[:ba, :bb, :bc] += lib.einsum('jbfa,kicf->abcijk',
                    eris_ovvv[:, b0:b1, :, a0:a1], t2[:, :, c0:c1, :])
                w_blk[:ba, :bb, :bc] += lib.einsum('jbfc,ikaf->abcijk',
                    eris_ovvv[:, b0:b1, :, c0:c1], t2[:, :, a0:a1, :])
                w_blk[:ba, :bb, :bc] += lib.einsum('kcfa,jibf->abcijk',
                    eris_ovvv[:, c0:c1, :, a0:a1], t2[:, :, b0:b1, :])
                w_blk[:ba, :bb, :bc] += lib.einsum('kcfb,ijaf->abcijk',
                    eris_ovvv[:, c0:c1, :, b0:b1], t2[:, :, a0:a1, :])
                w_blk[:ba, :bb, :bc] -= lib.einsum('iajm,mkbc->abcijk',
                    eris_ovoo[:, a0:a1, :, :], t2[:, :, b0:b1, c0:c1])
                w_blk[:ba, :bb, :bc] -= lib.einsum('iakm,mjcb->abcijk',
                    eris_ovoo[:, a0:a1, :, :], t2[:, :, c0:c1, b0:b1])
                w_blk[:ba, :bb, :bc] -= lib.einsum('jbim,mkac->abcijk',
                    eris_ovoo[:, b0:b1, :, :], t2[:, :, a0:a1, c0:c1])
                w_blk[:ba, :bb, :bc] -= lib.einsum('jbkm,mica->abcijk',
                    eris_ovoo[:, b0:b1, :, :], t2[:, :, c0:c1, a0:a1])
                w_blk[:ba, :bb, :bc] -= lib.einsum('kcim,mjab->abcijk',
                    eris_ovoo[:, c0:c1, :, :], t2[:, :, a0:a1, b0:b1])
                w_blk[:ba, :bb, :bc] -= lib.einsum('kcjm,miba->abcijk',
                    eris_ovoo[:, c0:c1, :, :], t2[:, :, b0:b1, a0:a1])

                v_blk[:ba, :bb, :bc] = lib.einsum('iajb,kc->abcijk',
                    eris_ovov[:, a0:a1, :, b0:b1], t1[:, c0:c1])
                v_blk[:ba, :bb, :bc] += lib.einsum('iakc,jb->abcijk',
                    eris_ovov[:, a0:a1, :, c0:c1], t1[:, b0:b1])
                v_blk[:ba, :bb, :bc] += lib. einsum('jbkc,ia->abcijk',
                    eris_ovov[:, b0:b1, :, c0:c1], t1[:, a0:a1])
                v_blk[:ba, :bb, :bc] += lib.einsum('ck,ijab->abcijk',
                    eris.fock[nocc + c0:nocc + c1, :nocc], t2[:, :, a0:a1, b0:b1])
                v_blk[:ba, :bb, :bc] += lib.einsum('ai,jkbc->abcijk',
                    eris.fock[nocc + a0:nocc + a1, :nocc], t2[:, :, b0:b1, c0:c1])
                v_blk[:ba, :bb, :bc] += lib.einsum('bj,kica->abcijk',
                    eris.fock[nocc + b0:nocc + b1, :nocc], t2[:, :, c0:c1, a0:a1])

                d3_blk = lib.direct_sum('ia,jb,kc->abcijk', eia[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1])
                w_blk[:ba, :bb, :bc] /= d3_blk
                v_blk[:ba, :bb, :bc] /= d3_blk

                v_blk += 2.0 * w_blk
                t3_symm_ip_py(v_blk, blksize**3, nocc, "2-1000-1", 1.0, 0.0)
                joovv[:, :, a0:a1, :] += lib.einsum('kceb,abcijk->ijae',
                    eris_ovvv[:, c0:c1, :, b0:b1], v_blk[:ba, :bb, :bc, :, :, :])
                joovv[:, :, a0:a1, b0:b1] -= lib.einsum('ncmj,abcimn->ijab',
                    eris_ovoo[:, c0:c1, :, :], v_blk[:ba, :bb, :bc, :, :, :])

                w_blk_r6 = numpy.copy(w_blk)
                t3_symm_ip_py(w_blk_r6, blksize**3, nocc, "4-2-211-2", 0.5, 0.0)
                imds.l1_t[:, a0:a1] += lib.einsum('jbkc,abcijk->ia',
                    eris_ovov[:, b0:b1, :, c0:c1], w_blk_r6[:ba, :bb, :bc, :, :, :])

                t3_symm_ip_py(w_blk, blksize**3, nocc, "2-1000-1", 0.5, 0.0)
                joovv[:, :, a0:a1, b0:b1] += lib.einsum('kc,abcijk->ijab',
                    eris.fock[:nocc, nocc + c0:nocc + c1], w_blk[:ba, :bb, :bc, :, :, :])

        time2 = log.timer_debug1('ccsd_t lambda make_intermediates [%d:%d]'%(c0, c1), *time2)

    w_blk = None
    v_blk = None

    imds.l1_t /= eia

    joovv = joovv + joovv.transpose(1, 0, 3, 2)
    imds.l2_t = joovv / lib.direct_sum('ia+jb->ijab', eia, eia)

    return imds

def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if eris is None: eris = mycc.ao2mo()
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = ccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 += imds.l1_t
    l2 += imds.l2_t
    return l1, l2

def t3_symm_ip_py(A, nocc3, nvir, pattern, alpha=1.0, beta=0.0):
    assert A.dtype == numpy.float64 and A.flags['C_CONTIGUOUS'], "A must be a contiguous float64 array"

    pattern_c = pattern.encode('utf-8')

    drv = _ccsd.libcc.t3_symm_ip
    drv(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc3),
        ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha),
        ctypes.c_double(beta)
    )
    return A


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-16
    rhf.scf()

    mcc = ccsd.CCSD(rhf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()
    #l1, l2 = mcc.solve_lambda()
    #print(numpy.linalg.norm(l1)-0.0132626841292)
    #print(numpy.linalg.norm(l2)-0.212575609057)

    conv, l1, l2 = kernel(mcc, mcc.ao2mo(), t1, t2, tol=1e-8)
    print(numpy.linalg.norm(l1)-0.013575484203926739)
    print(numpy.linalg.norm(l2)-0.22029981372536928)
