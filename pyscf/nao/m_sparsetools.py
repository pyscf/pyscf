# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
      print(x.size, ncol)
      raise ValueError("wrong dimension!")

    xx = np.require(x, requirements=["A", "O"])
    
    if csr.dtype == np.float32:
        y = np.zeros((nrow), dtype=np.float32)
        libsparsetools.scsr_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csr.indptr.ctypes.data_as(POINTER(c_int)),
                csr.indices.ctypes.data_as(POINTER(c_int)), 
                csr.data.ctypes.data_as(POINTER(c_float)),
                xx.ctypes.data_as(POINTER(c_float)), 
                y.ctypes.data_as(POINTER(c_float)))

    elif csr.dtype == np.float64:
        y = np.zeros((nrow), dtype=np.float64)
        libsparsetools.dcsr_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csr.indptr.ctypes.data_as(POINTER(c_int)),
                csr.indices.ctypes.data_as(POINTER(c_int)), 
                csr.data.ctypes.data_as(POINTER(c_double)),
                xx.ctypes.data_as(POINTER(c_double)), 
                y.ctypes.data_as(POINTER(c_double)))
    else:
        raise ValueError("Not implemented")

    return y


def csc_matvec(csc, x):
    """
        Matrix vector multiplication
        using csc format
    """

    if not sparse.isspmatrix_csc(csc):
        raise Exception("Matrix must be in csc format")

    nrow, ncol = csc.shape
    nnz = csc.data.shape[0]
    if x.size != ncol:
      print(x.size, ncol)
      raise ValueError("wrong dimension!")

    xx = np.require(x, requirements="C")

    if csc.dtype == np.float32:
        y = np.zeros((nrow), dtype=np.float32)
        libsparsetools.scsc_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csc.indptr.ctypes.data_as(POINTER(c_int)),
                csc.indices.ctypes.data_as(POINTER(c_int)), 
                csc.data.ctypes.data_as(POINTER(c_float)),
                xx.ctypes.data_as(POINTER(c_float)), 
                y.ctypes.data_as(POINTER(c_float)))

    elif csc.dtype == np.float64:
        y = np.zeros((nrow), dtype=np.float64)
        libsparsetools.dcsc_matvec(c_int(nrow), c_int(ncol), c_int(nnz), 
                csc.indptr.ctypes.data_as(POINTER(c_int)),
                csc.indices.ctypes.data_as(POINTER(c_int)), 
                csc.data.ctypes.data_as(POINTER(c_double)),
                xx.ctypes.data_as(POINTER(c_double)), 
                y.ctypes.data_as(POINTER(c_double)))
    else:
        raise ValueError("Not implemented")

    return y

def csc_matvecs(csc, B, transB = False, order="C"):
    """
        Matrix matrix multiplication
        using csc format
    """

    if not sparse.isspmatrix_csc(csc):
        raise Exception("Matrix must be in csc format")

    if transB:
        # Here need to be careful, since using the transpose of B
        # will change from row major to col major and vice-versa
        mat = np.require(B.T, dtype=B.dtype, requirements=["A", "O", order])
    else:
        mat = np.require(B, dtype=B.dtype, requirements=["A", "O", order])

    nrow, ncol = csc.shape
    nvecs = mat.shape[1]

    if csc.dtype == np.float32:
        C = np.zeros((nrow, nvecs), dtype=np.float32, order=order)
        libsparsetools.scsc_matvecs(c_int(nrow), c_int(ncol), c_int(nvecs), 
                csc.indptr.ctypes.data_as(POINTER(c_int)),
                csc.indices.ctypes.data_as(POINTER(c_int)), 
                csc.data.ctypes.data_as(POINTER(c_float)),
                mat.ctypes.data_as(POINTER(c_float)), 
                C.ctypes.data_as(POINTER(c_float)))

    elif csc.dtype == np.float64:
        C = np.zeros((nrow, nvecs), dtype=np.float64, order=order)
        libsparsetools.dcsc_matvecs(c_int(nrow), c_int(ncol), c_int(nvecs), 
                csc.indptr.ctypes.data_as(POINTER(c_int)),
                csc.indices.ctypes.data_as(POINTER(c_int)), 
                csc.data.ctypes.data_as(POINTER(c_double)),
                mat.ctypes.data_as(POINTER(c_double)), 
                C.ctypes.data_as(POINTER(c_double)))
    else:
        raise ValueError("Not implemented")

    return C


