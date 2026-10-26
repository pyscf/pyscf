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
from pyscf.cc import ccsd, ccsd_lambda, ccsd_t_rdm, _ccsd

# Note: not support fov != 0

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)

    if numpy.iscomplexobj(t1) or numpy.iscomplexobj(t2) or numpy.iscomplexobj(eris):
        raise ValueError("make_intermediates does not support complex-valued inputs (t1, t2, or eris)")

    time0 = logger.process_clock(), logger.perf_counter()
    imds = ccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    mo_e, t1T, t2T, vooo, vvop, fvo, fov = ccsd_t_rdm._t3_kernel_args(t1, t2, eris)
    l1_t = numpy.zeros((nocc, nvir))
    joovv = numpy.zeros((nocc, nocc, nvir, nvir))
    drv = _ccsd.libcc.CCsd_t_lambda_intermediates
    drv.restype = ctypes.c_int
    err = drv(ctypes.c_int(nocc), ctypes.c_int(nvir),
              mo_e.ctypes.data_as(ctypes.c_void_p), t1T.ctypes.data_as(ctypes.c_void_p),
              t2T.ctypes.data_as(ctypes.c_void_p), vooo.ctypes.data_as(ctypes.c_void_p),
              vvop.ctypes.data_as(ctypes.c_void_p), fvo.ctypes.data_as(ctypes.c_void_p),
              fov.ctypes.data_as(ctypes.c_void_p), l1_t.ctypes.data_as(ctypes.c_void_p),
              joovv.ctypes.data_as(ctypes.c_void_p))
    if err:
        raise MemoryError('CCsd_t_lambda_intermediates: failed to allocate thread buffers')
    vvop = t2T = None
    log.timer_debug1('ccsd_t lambda make_intermediates (T) part', *time0)

    eia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
    imds.l1_t = l1_t / eia
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
