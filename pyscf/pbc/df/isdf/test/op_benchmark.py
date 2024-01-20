
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
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, name))

@delayed
def _cwise_mul(a, b):
    return a * b

if __name__ == "__main__":

    # NORB   = 256
    # NGRID  = 5120  # c=20
    # aoRg    = np.random.random((NORB, NGRID))
    # J      = np.random.random((NGRID))
    # t1 = (logger.process_clock(), logger.perf_counter())
    # J = np.einsum('ij,j->ij', aoRg, J)
    # t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, 'getaopair')
    # t1 = (logger.process_clock(), logger.perf_counter())
    # rho_mu_nu_Rg = np.einsum('ij,kj->ikj', aoRg, aoRg)   # this could be very costly !
    # t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, 'getrho')
    # rho_mu_nu_Rg = None
    # NORB      = 64
    # NGRID_ALL = 46646
    # aoR       = np.random.random((NORB, NGRID_ALL))
    # tmp1      = np.random.random((NORB, NGRID_ALL))
    # t1 = (logger.process_clock(), logger.perf_counter())
    # density_R = np.einsum('ik,ik->k', aoR, tmp1)
    # t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, 'getdensity')
    # tmp1 = None
    # J = np.random.random((NGRID_ALL))
    # t1 = (logger.process_clock(), logger.perf_counter())
    # J = np.einsum('ij,j->ij', aoR, J)
    # t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, 'getaopair')
    # t1 = (logger.process_clock(), logger.perf_counter())
    # J = aoR * J
    # # print(J.shape)
    # t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, 'getaopair')
    # J = None
    # aoR = None
    # NGRID_ALL = 30000
    # V_R = np.random.random((NGRID, NGRID_ALL))
    # density_RgR = np.random.random((NGRID, NGRID_ALL))
    # t1 = (logger.process_clock(), logger.perf_counter())
    # W = numpy.multiply(V_R, density_RgR)
    # t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, 'get_K_kernel')
    # t1 = (logger.process_clock(), logger.perf_counter())
    # W = lib.cwise_mul(V_R, density_RgR)
    # t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, 'get_K_kernel')

    NORB   = 2560

    matrix = np.random.random((NORB, NORB))
    matrix += matrix.T

    t1 = (logger.process_clock(), logger.perf_counter())
    # e, h = np.linalg.eigh(matrix)
    # q, r = np.linalg.qr(matrix)
    e, h = scipy.linalg.eigh(matrix)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, 'eigh')
