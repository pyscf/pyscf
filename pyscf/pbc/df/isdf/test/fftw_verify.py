
import copy
from functools import reduce
import numpy as np
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_ao2mo as isdf_ao2mo
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk

import sys
import ctypes
import _ctypes

from multiprocessing import Pool

import dask.array as da
from dask import delayed

libfft = lib.load_library('libfft')

def _rfft3d(a, out=None, func="_rfft_3d_ISDF_manydft"):

    assert a.ndim >= 3

    if a.dtype != numpy.double:
        raise NotImplementedError
    else:
        fn = getattr(libfft, func, None)
        assert(fn is not None)

    mesh = numpy.array(a.shape[-3:], dtype=numpy.int32)

    if a.ndim > 3:
        ntrans = np.prod(a.shape[:-3], dtype=np.int32)
    else:
        ntrans = 1

    if out is None:
        out = numpy.empty((ntrans, mesh[0], mesh[1], mesh[2]//2+1), dtype=numpy.complex128)

    fn(a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p),
       mesh.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(ntrans))

    return out

def _irfft3d(a, mesh, out=None, func="_irfft_3d_ISDF_manydft"):

    assert a.ndim >= 3

    if a.dtype != numpy.complex128:
        raise NotImplementedError
    else:
        fn = getattr(libfft, func, None)
        assert(fn is not None)

    if a.ndim > 3:
        ntrans = np.prod(a.shape[:-3], dtype=np.int32)
    else:
        ntrans = 1

    mesh = numpy.array(mesh, dtype=numpy.int32)

    if out is None:
        out = numpy.empty((ntrans, mesh[0], mesh[1], mesh[2]), dtype=numpy.double)

    fn(a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p),
       mesh.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(ntrans))

    return out

if __name__ == "__main__":

    MESH = [
        [30, 30, 30],
        [50, 50, 50],
        [70, 70, 70],
        # [128, 128, 128],
        # [256, 256, 256]
    ]

    ## verify whether the fft is correct

    nTrans = 6

    a = np.random.rand(nTrans, 15, 15, 15)

    b = np.fft.rfftn(a, axes=(-3, -2, -1))
    c = np.fft.irfftn(b, a.shape[1:], axes=(-3, -2, -1))

    print(np.allclose(a, c))

    bb = _rfft3d(a)
    cc = _irfft3d(bb, a.shape[1:])

    print(np.allclose(a, cc))
    print(np.allclose(b, bb))

    bbb = _rfft3d(a, func="_rfft_3d_ISDF")
    ccc = _irfft3d(bbb, a.shape[1:], func="_irfft_3d_ISDF")

    print(np.allclose(a, ccc))
    print(np.allclose(b, bbb))

    bbbb = _rfft3d(a, func="_rfft_3d_ISDF_parallel")
    cccc = _irfft3d(bbbb, a.shape[1:], func="_irfft_3d_ISDF_parallel")

    print(np.allclose(a, cccc))
    print(np.allclose(b, bbbb))

    ## test the performance 

    for mesh in MESH:
        print("mesh = ", mesh)
        for ntrans in [1, 2, 4, 8, 16, 32, 64, 128]:
            
            print("ntrans = ", ntrans)

            a = np.random.rand(ntrans, *mesh)
        
            t1 = (logger.process_clock(), logger.perf_counter())
            b1 = np.fft.rfftn(a, axes=(-3, -2, -1))
            c1 = np.fft.irfftn(b1, a.shape[1:], axes=(-3, -2, -1))
            t2 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t1, t2, 'numpy fftn')

            t1 = (logger.process_clock(), logger.perf_counter())
            b2 = _rfft3d(a, func="_rfft_3d_ISDF_manydft")
            c2 = _irfft3d(b2, a.shape[1:], func="_irfft_3d_ISDF_manydft")
            t2 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t1, t2, 'fftw fft manydft')

            t1 = (logger.process_clock(), logger.perf_counter())
            b3 = _rfft3d(a, func="_rfft_3d_ISDF")
            c3 = _irfft3d(b3, a.shape[1:], func="_irfft_3d_ISDF")
            t2 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t1, t2, 'fftw fft')

            t1 = (logger.process_clock(), logger.perf_counter())
            b4 = _rfft3d(a, func="_rfft_3d_ISDF_parallel")
            c4 = _irfft3d(b4, a.shape[1:], func="_irfft_3d_ISDF_parallel")
            t2 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t1, t2, 'fftw fft parallel')