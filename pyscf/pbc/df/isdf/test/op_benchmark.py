
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


if __name__ == "__main__":
   
    t1 = (logger.process_clock(), logger.perf_counter()) 
    NORB   = 128
    NGRID  = 50000
    aoR    = np.random.random((NORB, NGRID))
    aoPair = np.einsum('ik,jk->ijk', aoR, aoR).reshape(-1, NGRID)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, 'getaopair')

    t1 = (logger.process_clock(), logger.perf_counter()) 
    NORB   = 128
    NGRID  = 2560
    aoR    = np.random.random((NORB, NGRID))
    aoPair = np.einsum('ik,jk->ijk', aoR, aoR).reshape(-1, NGRID)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, 'getaopair')

    t1 = (logger.process_clock(), logger.perf_counter())
    NORB2 = 32
    Mat = np.random.random((NORB2, NORB))
    Mat1 = Mat @ aoR
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, 'matvec')