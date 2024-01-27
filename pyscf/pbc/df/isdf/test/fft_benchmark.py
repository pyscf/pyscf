
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

libpbc = lib.load_library('libpbc')
libfft = lib.load_library('libfft')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, name))

def _rfft(a, out=None):

    if a.dtype != numpy.double:
        raise NotImplementedError
    else:
        fn = getattr(libfft, "rfft", None)
        assert(fn is not None)

    if out is None:
        out = numpy.empty((a.shape[0], a.shape[1], a.shape[2]//2+1), dtype=numpy.complex128)

    rank = len(a.shape)
    mesh = numpy.array(a.shape, dtype=numpy.int32)

    fn(a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p),
       mesh.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(rank))

    return out

def _irfft(a, mesh, out=None):

    if a.dtype != numpy.complex128:
        raise NotImplementedError
    else:
        fn = getattr(libfft, "irfft", None)
        assert(fn is not None)

    if out is None:
        out = numpy.empty(mesh, dtype=numpy.double)

    # mesh = np.asarray(mesh, dtype=np.int32)
    mesh = numpy.array(mesh, dtype=numpy.int32)
    rank = len(a.shape)

    fn(a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p),
       mesh.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(rank))

    return out

def _rfft3d(a, out=None):

    assert a.ndim == 3

    if a.dtype != numpy.double:
        raise NotImplementedError
    else:
        fn = getattr(libfft, "rfft_3d", None)
        assert(fn is not None)

    if out is None:
        out = numpy.empty((a.shape[0], a.shape[1], a.shape[2]//2+1), dtype=numpy.complex128)

    rank = len(a.shape)
    mesh = numpy.array(a.shape, dtype=numpy.int32)

    fn(a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p),
       mesh.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(rank))

    return out

def _irfft3d(a, mesh, out=None):

    assert a.ndim == 3

    if a.dtype != numpy.complex128:
        raise NotImplementedError
    else:
        fn = getattr(libfft, "irfft_3d", None)
        assert(fn is not None)

    if out is None:
        out = numpy.empty(mesh, dtype=numpy.double)

    mesh = numpy.array(mesh, dtype=numpy.int32)
    rank = len(a.shape)

    fn(a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p),
       mesh.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(rank))

    return out

if __name__ == "__main__":

    MESH = [
        [100, 100, 100],
        [3, 3, 3],
        [30, 30, 30],
        [50, 50, 50],
        [70, 70, 70],
        [128, 128, 128],
        [256, 256, 256]
        # [3, 3, 3],
        # [4, 4, 4],
    ]

    for mesh in MESH:

        print("mesh = ", mesh)

        a = numpy.random.random((mesh[0], mesh[1], mesh[2]))

        t1 = (logger.process_clock(), logger.perf_counter())
        b = numpy.fft.rfftn(a)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'numpy_rfft')

        t1 = (logger.process_clock(), logger.perf_counter())
        c = _rfft(a)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'fftw_rfft')

        t1 = (logger.process_clock(), logger.perf_counter())
        d = _rfft3d(a)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'fftw_rfft3d')

        t1 = (logger.process_clock(), logger.perf_counter())
        bb = numpy.fft.fftn(a)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'numpy_fftn')

        # print(c.shape)

        t1 = (logger.process_clock(), logger.perf_counter())
        e = numpy.fft.irfftn(b, a.shape)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'numpy_irfft')

        t1 = (logger.process_clock(), logger.perf_counter())
        f = _irfft(c, a.shape)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'fftw_irfft')

        t1 = (logger.process_clock(), logger.perf_counter())
        g = _irfft(d, a.shape)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'fftw_irfft3d')

        t1 = (logger.process_clock(), logger.perf_counter())
        cc = numpy.fft.ifftn(bb) # extremly slow for large mesh
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'numpy_ifftn')

        t1 = (logger.process_clock(), logger.perf_counter())
        print(numpy.allclose(c, b))
        print(numpy.allclose(d, b))
        print(numpy.allclose(a, f))
        print(numpy.allclose(a, e))
        print(numpy.allclose(a, g))
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, 'check')

        print(a.flags)
        print(b.flags)
        print(c.flags)
        print(d.flags)
        print(e.flags)
        print(f.flags)
        print(g.flags)
        print(bb.flags)
        print(cc.flags)

    # print("mesh = ", mesh)