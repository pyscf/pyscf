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

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_rdm, _ccsd

def _t3_kernel_args(t1, t2, eris):
    '''Sorted inputs of the (T) kernels in libcc (ccsd_t_rdm.c), active space.'''
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovov = numpy.asarray(eris.ovov)
    vvop = numpy.empty((nvir, nvir, nocc, nmo))
    vvop[:, :, :, :nocc] = eris_ovov.transpose(1, 3, 0, 2)
    vvop[:, :, :, nocc:] = eris_ovvv.transpose(1, 3, 0, 2)
    eris_ovvv = eris_ovov = None
    fock = numpy.asarray(eris.fock)
    return (numpy.asarray(eris.mo_energy, dtype=numpy.float64, order='C'),
            numpy.asarray(t1.T, dtype=numpy.float64, order='C'),
            numpy.asarray(t2.transpose(2, 3, 1, 0), dtype=numpy.float64, order='C'),
            numpy.asarray(numpy.asarray(eris.ovoo).transpose(1, 0, 2, 3), dtype=numpy.float64, order='C'),
            vvop,
            numpy.asarray(fock[nocc:, :nocc], dtype=numpy.float64, order='C'),
            numpy.asarray(fock[:nocc, nocc:], dtype=numpy.float64, order='C'))

def _t3_intermediates(t1, t2, eris, with_gamma1=True, with_gamma2=True):
    '''(T) contributions goo, gvv, dvo, dovov, dooov, dovvv of the CCSD(T) response
    density intermediates (libcc CCsd_t_rdm_intermediates; one pass over the virtual
    triples).  The arrays of a skipped gamma are returned as zeros.
    '''
    nocc, nvir = t1.shape
    mo_e, t1T, t2T, vooo, vvop, fvo = _t3_kernel_args(t1, t2, eris)[:6]
    goo = numpy.zeros((nocc, nocc))
    gvv = numpy.zeros((nvir, nvir))
    dvo = numpy.zeros((nvir, nocc))
    dovov = numpy.zeros((nocc, nvir, nocc, nvir))
    dooov = numpy.zeros((nocc, nocc, nocc, nvir))
    dovvv = numpy.zeros((nocc, nvir, nvir, nvir))
    drv = _ccsd.libcc.CCsd_t_rdm_intermediates
    drv.restype = ctypes.c_int
    err = drv(ctypes.c_int(nocc), ctypes.c_int(nvir),
              mo_e.ctypes.data_as(ctypes.c_void_p), t1T.ctypes.data_as(ctypes.c_void_p),
              t2T.ctypes.data_as(ctypes.c_void_p), vooo.ctypes.data_as(ctypes.c_void_p),
              vvop.ctypes.data_as(ctypes.c_void_p), fvo.ctypes.data_as(ctypes.c_void_p),
              goo.ctypes.data_as(ctypes.c_void_p), gvv.ctypes.data_as(ctypes.c_void_p),
              dvo.ctypes.data_as(ctypes.c_void_p), dovov.ctypes.data_as(ctypes.c_void_p),
              dooov.ctypes.data_as(ctypes.c_void_p), dovvv.ctypes.data_as(ctypes.c_void_p),
              ctypes.c_int(with_gamma1), ctypes.c_int(with_gamma2))
    if err:
        raise MemoryError('CCsd_t_rdm_intermediates: failed to allocate thread buffers')
    return goo, gvv, dvo, dovov, dooov, dovvv

def _gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None, for_grad=False):
    log = logger.Logger(mycc.stdout, mycc.verbose)

    if (numpy.iscomplexobj(t1) or numpy.iscomplexobj(t2) or numpy.iscomplexobj(eris)
        or numpy.iscomplexobj(l1) or numpy.iscomplexobj(l2)):
        raise ValueError("_gamma1_intermediates does not support complex-valued inputs (t1, t2, l1, l2, or eris)")

    doo, dov, dvo, dvv = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None: eris = mycc.ao2mo()
    nocc, nvir = t1.shape
    time0 = logger.process_clock(), logger.perf_counter()
    goo, gvv, dvo_t = _t3_intermediates(t1, t2, eris, with_gamma2=False)[:3]
    dvo = dvo + dvo_t
    log.timer_debug1("ccsd_t rdm _gamma1_intermediates (T) part", *time0)

    if not for_grad:
        # t3 amplitudes in CCSD(T) is computed non-iteratively. The
        # off-diagonal blocks of fock matrix does not contribute to CCSD(T)
        # energy. To make Tr(H,D) consistent to the CCSD(T) total energy, the
        # density matrix off-diagonal parts are excluded.
        doo[numpy.diag_indices(nocc)] -= goo.diagonal() * .5
        dvv[numpy.diag_indices(nvir)] += gvv.diagonal() * .5

    else:
        # The off-diagonal blocks of fock matrix have small contributions to
        # analytical nuclear gradients.
        doo -= goo * .5
        dvv += gvv * .5

    return doo, dov, dvo, dvv

def _gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None,
                          compress_vvvv=False):
    '''intermediates tensors for gamma2 are sorted in Chemist's notation
    '''
    log = logger.Logger(mycc.stdout, mycc.verbose)

    if (numpy.iscomplexobj(t1) or numpy.iscomplexobj(t2) or numpy.iscomplexobj(eris)
        or numpy.iscomplexobj(l1) or numpy.iscomplexobj(l2)):
        raise ValueError("_gamma2_intermediates does not support complex-valued inputs (t1, t2, l1, l2, or eris)")

    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            ccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
    if eris is None: eris = mycc.ao2mo()

    time0 = logger.process_clock(), logger.perf_counter()
    dovov_t, dooov_t, dovvv_t = _t3_intermediates(t1, t2, eris, with_gamma1=False)[3:]
    log.timer_debug1("ccsd_t rdm _gamma2_intermediates (T) part", *time0)
    dovov = dovov + dovov_t
    dooov = dooov + dooov_t
    dovvv = dovvv + dovvv_t

    dvvov = dovvv.transpose(2,3,0,1)

    if compress_vvvv:
        nvir = mycc.nmo - mycc.nocc
        idx = numpy.tril_indices(nvir)
        vidx = idx[0] * nvir + idx[1]
        dvvvv = dvvvv + dvvvv.transpose(1,0,2,3)
        dvvvv = dvvvv + dvvvv.transpose(0,1,3,2)
        dvvvv = lib.take_2d(dvvvv.reshape(nvir**2,nvir**2), vidx, vidx)
        dvvvv *= .25

    return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov

def _gamma2_outcore(mycc, t1, t2, l1, l2, eris, h5fobj, compress_vvvv=False):
    return _gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv)

def make_rdm1(mycc, t1, t2, l1, l2, eris=None, ao_repr=False):
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    return ccsd_rdm._make_rdm1(mycc, d1, True, ao_repr=ao_repr)

# rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, eris=None):
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    return ccsd_rdm._make_rdm2(mycc, d1, d2, True, True)

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
