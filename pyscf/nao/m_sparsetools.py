from pyscf.lib import misc
import numpy as np
import scipy.sparse as sparse
from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int

libsparsetools = misc.load_library("libsparsetools")

"""
    Wrapper to sparse matrix operations from scipy implemented with openmp
"""
def csr_matvec(csr, x):

    if not sparse.isspmatrix_csr(csr):
        raise Exception("Matrix must be in csr format")

    nrow, ncol = csr.shape
    nnz = csr.data.shape[0]
    if x.size != ncol:
        raise ValueError("wrong dimension!")

    if csr.dtype == np.float32:
        y = np.zeros((nrow), dtype=np.float32)
        libsparsetools.scsr_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csr.indptr.ctypes.data_as(POINTER(c_int)),
                csr.indices.ctypes.data_as(POINTER(c_int)), 
                csr.data.ctypes.data_as(POINTER(c_float)),
                x.ctypes.data_as(POINTER(c_float)), 
                y.ctypes.data_as(POINTER(c_float)))

    elif csr.dtype == np.float64:
        y = np.zeros((nrow), dtype=np.float64)
        libsparsetools.dcsr_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csr.indptr.ctypes.data_as(POINTER(c_int)),
                csr.indices.ctypes.data_as(POINTER(c_int)), 
                csr.data.ctypes.data_as(POINTER(c_double)),
                x.ctypes.data_as(POINTER(c_double)), 
                y.ctypes.data_as(POINTER(c_double)))
    else:
        raise ValueError("Not implemented")

    return y


def csc_matvec(csc, x):

    if not sparse.isspmatrix_csc(csc):
        raise Exception("Matrix must be in csc format")

    nrow, ncol = csc.shape
    nnz = csc.data.shape[0]
    if x.size != ncol:
        raise ValueError("wrong dimension!")

    if csc.dtype == np.float32:
        y = np.zeros((nrow), dtype=np.float32)
        libsparsetools.scsc_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csc.indptr.ctypes.data_as(POINTER(c_int)),
                csc.indices.ctypes.data_as(POINTER(c_int)), 
                csc.data.ctypes.data_as(POINTER(c_float)),
                x.ctypes.data_as(POINTER(c_float)), 
                y.ctypes.data_as(POINTER(c_float)))

    elif csc.dtype == np.float64:
        y = np.zeros((nrow), dtype=np.float64)
        libsparsetools.dcsc_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csc.indptr.ctypes.data_as(POINTER(c_int)),
                csc.indices.ctypes.data_as(POINTER(c_int)), 
                csc.data.ctypes.data_as(POINTER(c_double)),
                x.ctypes.data_as(POINTER(c_double)), 
                y.ctypes.data_as(POINTER(c_double)))
    else:
        raise ValueError("Not implemented")

    return y
