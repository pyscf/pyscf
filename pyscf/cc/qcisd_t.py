#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
RHF-QCISD(T) for real integrals
'''


import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import _ccsd
from pyscf.cc.ccsd_t import _sort_eri, _sort_t2_vooo_

# t3 as ijkabc

# JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    cpu1 = cpu0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    dtype = numpy.result_type(t1, t2, eris.ovoo.dtype)
    if mycc.incore_complete:
        ftmp = None
        eris_vvop = numpy.zeros((nvir,nvir,nocc,nmo), dtype)
    else:
        ftmp = lib.H5TmpFile()
        eris_vvop = ftmp.create_dataset('vvop', (nvir,nvir,nocc,nmo), dtype)

    orbsym = _sort_eri(mycc, eris, nocc, nvir, eris_vvop, log)

    mo_energy, t1T, t2T, vooo, fvo, restore_t2_inplace = \
            _sort_t2_vooo_(mycc, orbsym, t1, t2, eris)
    cpu1 = log.timer_debug1('QCISD(T) sort_eri', *cpu1)

    cpu2 = list(cpu1)
    orbsym = numpy.hstack((numpy.sort(orbsym[:nocc]),numpy.sort(orbsym[nocc:])))
    o_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[:nocc], minlength=8)))
    v_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[nocc:], minlength=8)))
    o_sym = orbsym[:nocc]
    oo_sym = (o_sym[:,None] ^ o_sym).ravel()
    oo_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(oo_sym, minlength=8)))
    nirrep = max(oo_sym) + 1

    orbsym   = orbsym.astype(numpy.int32)
    o_ir_loc = o_ir_loc.astype(numpy.int32)
    v_ir_loc = v_ir_loc.astype(numpy.int32)
    oo_ir_loc = oo_ir_loc.astype(numpy.int32)
    if dtype == numpy.complex128:
        drv = _ccsd.libcc.QCIsd_t_zcontract
    else:
        drv = _ccsd.libcc.QCIsd_t_contract
    et_sum = numpy.zeros(1, dtype=dtype)
    def contract(a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(nirrep),
            o_ir_loc.ctypes.data_as(ctypes.c_void_p),
            v_ir_loc.ctypes.data_as(ctypes.c_void_p),
            oo_ir_loc.ctypes.data_as(ctypes.c_void_p),
            orbsym.ctypes.data_as(ctypes.c_void_p),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)

    # The rest 20% memory for cache b
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    bufsize = (max_memory*.5e6/8-nocc**3*3*lib.num_threads())/(nocc*nmo)  #*.5 for async_io
    bufsize *= .5  #*.5 upper triangular part is loaded
    bufsize *= .8  #*.8 for [a0:a1]/[b0:b1] partition
    bufsize = max(8, bufsize)
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    with lib.call_in_background(contract, sync=not mycc.async_io) as async_contract:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvir, bufsize))):
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1], order='C')
            async_contract(a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                            cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/8):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                async_contract(a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                                cache_row_b,cache_col_b))

    t2 = restore_t2_inplace(t2T)
    et_sum *= 2
    if abs(et_sum[0].imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part of QCISD(T) energy was found %s',
                    et_sum[0])
    et = et_sum[0].real
    log.timer('QCISD(T)', *cpu0)
    log.note('QCISD(T) correction = %.15g', et)
    return et
