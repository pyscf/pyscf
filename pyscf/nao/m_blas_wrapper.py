from __future__ import division
import numpy as np
from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int
from pyscf.nao.m_libnao import libnao

"""
Implemented small blas wrapper that are missing in scipy.linalg.blas
"""

def spmv_wrapper(alpha, ap, x, beta = 0.0, incx = 1, incy = 1, uplo="U"):

    n = x.size
    if ap.size != int(n*(n+1)/2):
        raise ValueError("simple wrapper, you MUST provide x.size = n, ap.size = n*(n+1)/2")
    
    if uplo not in ["L", "U"]:
        raise ValueError("uplo must be L or U")

    # uplo must be inversed if using row major
    uplo_dict = {"U": 1, "L": 0}

    if ap.dtype == np.float32:
        y = np.zeros((n), dtype=np.float32)
        libnao.SSPMV_wrapper(c_int(uplo_dict[uplo]), c_int(n), c_float(alpha),
                ap.ctypes.data_as(POINTER(c_float)),
                x.ctypes.data_as(POINTER(c_float)), c_int(incx), c_float(beta),
                y.ctypes.data_as(POINTER(c_float)), c_int(incy))
    elif ap.dtype == np.float64:
        y = np.zeros((n), dtype=np.float64)
        libnao.DSPMV_wrapper(c_int(uplo_dict[uplo]), c_int(n), c_double(alpha),
                ap.ctypes.data_as(POINTER(c_double)),
                x.ctypes.data_as(POINTER(c_double)), c_int(incx), c_double(beta),
                y.ctypes.data_as(POINTER(c_double)), c_int(incy))
    else:
        raise ValueError("dtype error, only np.float32 and np.float64 implemented")

    return y
