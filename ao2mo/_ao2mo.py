#!/usr/bin/env python

import os
import ctypes
import _ctypes
import numpy
import pyscf.lib

libao2mo = pyscf.lib.load_library('libao2mo')


def _count_ij(istart, icount, jstart, jcount):
    if jstart+jcount <= istart:
        ntri = 0
    else:
        noff = jstart+jcount - (istart + 1)
        ntri = noff*(noff+1)/2
    return icount*jcount - ntri

def _get_num_threads():
    libao2mo.omp_get_num_threads.restype = ctypes.c_int
    nthreads = libao2mo.omp_get_num_threads()
    return nthreads

def nr_e1(mo_coeff, shape, atm, bas, env, vout=None):
    mo_coeff = numpy.asfortranarray(mo_coeff)
    i0, ic, j0, jc = shape
    assert(j0 <= i0)
    assert(j0+jc <= i0+ic <= mo_coeff.shape[1])
    nao = mo_coeff.shape[0]
    nao_pair = nao*(nao+1)/2

    if vout is None:
        vout = numpy.empty((_count_ij(*shape),nao_pair))
    if ic < jc:
        if i0+70 > j0+jc:
            fn = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                               'AO2MOnr_tri_e1_o4'))
        else:
            fn = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                               'AO2MOnr_tri_e1_o2'))
    else:
        if i0+70 > j0+jc:
            fn = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                               'AO2MOnr_tri_e1_o3'))
        else:
            fn = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                               'AO2MOnr_tri_e1_o1'))

    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    libao2mo.AO2MOnr_e1direct_drv(vout.ctypes.data_as(ctypes.c_void_p),
                                  mo_coeff.ctypes.data_as(ctypes.c_void_p), fn,
                                  ctypes.c_int(i0), ctypes.c_int(ic),
                                  ctypes.c_int(j0), ctypes.c_int(jc),
                                  c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                                  c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                  c_env.ctypes.data_as(ctypes.c_void_p))
    return vout

# in-place transform AO to MO
def nr_e2(eri, mo_coeff, shape, vout=None):
    assert(eri.flags.c_contiguous)
    mo_coeff = numpy.asfortranarray(mo_coeff)
    i0, ic, j0, jc = shape
    assert(j0 <= i0)
    assert(j0+jc <= i0+ic <= mo_coeff.shape[1])
    nao = mo_coeff.shape[0]
    nao_pair = nao*(nao+1)/2
    nrow = eri.shape[0]
    nij = _count_ij(*shape)

    if vout is None:
        if nij == nao_pair: # we can reuse memory
            vout = eri
        else:
# memory cannot be reused unless nij == nao_pair, even nij < nao_pair.
# When OMP is used, data is not accessed sequentially, the transformed
# integrals can overwrite the AO integrals which are not used.
            vout = numpy.empty((nrow,nij))

    if ic < jc:
        fn = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                           'AO2MOnr_tri_e2_o2'))
    else:
        fn = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                           'AO2MOnr_tri_e2_o1'))

    libao2mo.AO2MOnr_e2_drv(ctypes.c_int(nrow),
                            vout.ctypes.data_as(ctypes.c_void_p),
                            eri.ctypes.data_as(ctypes.c_void_p),
                            mo_coeff.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(nao), fn,
                            ctypes.c_int(i0), ctypes.c_int(ic),
                            ctypes.c_int(j0), ctypes.c_int(jc))
    return vout

def nr_e1range(mo_coeff, sh_range, shape, atm, bas, env, vout=None):
    mo_coeff = numpy.asfortranarray(mo_coeff)
    i0, ic, j0, jc = shape
    assert(j0 <= i0)
    assert(j0+jc <= i0+ic <= mo_coeff.shape[1])

    if ic <= jc:
        fn = libao2mo.AO2MOnr_e1range_o2
    else:
        fn = libao2mo.AO2MOnr_e1range_o1

    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    ish0, ish1, buflen = sh_range

    if vout is None:
        vout = numpy.empty((buflen,_count_ij(*shape)))
    libao2mo.AO2MOnr_e1outcore_drv(vout.ctypes.data_as(ctypes.c_void_p),
                                   mo_coeff.ctypes.data_as(ctypes.c_void_p),
                                   fn,
                                   ctypes.c_int(ish0), ctypes.c_int(ish1),
                                   ctypes.c_int(i0), ctypes.c_int(ic),
                                   ctypes.c_int(j0), ctypes.c_int(jc),
                                   c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                                   c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                   c_env.ctypes.data_as(ctypes.c_void_p))
    return vout


def nr_e1_incore(eri_ao, mo_coeff, shape, vout=None):
    assert(eri_ao.flags.c_contiguous)
    mo_coeff = numpy.asfortranarray(mo_coeff)
    i0, ic, j0, jc = shape
    assert(j0 <= i0)
    assert(j0+jc <= i0+ic <= mo_coeff.shape[1])

    nao = mo_coeff.shape[0]
    nao_pair = nao*(nao+1)/2

    if vout is None:
        vout = numpy.empty((_count_ij(*shape),nao_pair))
    if eri_ao.size != nao_pair**2: # 8-fold symmetry of eri_ao
        facc = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                             'AO2MOnr_incore8f_acc'))
    else: # 4-fold symmetry of eri_ao
        facc = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                             'AO2MOnr_incore4f_acc'))
    if ic <= jc:
        ftrans = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                               'AO2MOnr_tri_e2_o2'))
    else:
        ftrans = ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle,
                                               'AO2MOnr_tri_e2_o1'))
    libao2mo.AO2MOnr_e1incore_drv(vout.ctypes.data_as(ctypes.c_void_p),
                                  eri_ao.ctypes.data_as(ctypes.c_void_p),
                                  mo_coeff.ctypes.data_as(ctypes.c_void_p),
                                  facc, ftrans, ctypes.c_int(nao),
                                  ctypes.c_int(i0), ctypes.c_int(ic),
                                  ctypes.c_int(j0), ctypes.c_int(jc))
    return vout

