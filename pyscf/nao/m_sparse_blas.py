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
from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int
import scipy.sparse as sparse
import numpy as np

try:
  libspblas = misc.load_library("libsparse_blas")
  use_sparse_blas = True
except:
  #raise UserWarning("Using scipy version")
  libspblas = None
  use_sparse_blas = False 
 
 
#def csrmm(alpha, csr, B, beta = 0.0, trans="N", colrow = "row"):
#    """
#        Small wrapper to use the mkl_cspblas_?csrgemv routine
#        instead of the scipy.sparse mul operation.
#    Inputs:
#    -------
#        csr: scipy csr matrix
#        x: 1D numpy array of size N
#        trans: string for trans csr, N or T
#    
#    Outputs:
#    --------
#        y: 1D numpy array of size N
#    """
#
#    if not sparse.isspmatrix_csr(csr):
#        raise Exception("Matrix must be in csr format")
#    if use_sparse_blas:
#        trans_dico = {"N": 0, "T": 1}
#
#        if trans == "N":
#            m = csr.shape[0]
#            n = B.shape[1]
#            k = csr.shape[1]
#
#        else:
#            raise ValueError("Not implemented")
#            
#        if csr.dtype == np.float32 and B.dtype = np.float32:
#
#            libspblas.mkl_scsrmm()
#
#    
#    else:
#        return csr*B
  
def csrgemv(csr, x, trans="N", colrow = "row"):
    """
        Small wrapper to use the mkl_cspblas_?csrgemv routine
        instead of the scipy.sparse mul operation.
    Inputs:
    -------
        csr: scipy csr matrix
        x: 1D numpy array of size N
        trans: string for trans csr, N or T
    
    Outputs:
    --------
        y: 1D numpy array of size N
    """

    if not sparse.isspmatrix_csr(csr):
        raise Exception("Matrix must be in csr format")

    if use_sparse_blas:
        trans_dioc = {"N": 0, "T": 1}
        if colrow == "row":
            nrow = csr.shape[0]
        else:
            #trans_dioc = {"N": 1, "T": 0}
            nrow = csr.shape[1]

        if csr.data.dtype == np.float32 and x.dtype == np.float32:
            y = np.zeros((nrow), dtype=np.float32)
            libspblas.SCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    csr.data.ctypes.data_as(POINTER(c_float)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    x.ctypes.data_as(POINTER(c_float)), y.ctypes.data_as(POINTER(c_float)))

        elif csr.data.dtype == np.float32 and x.dtype == np.complex64:
            y_real = np.zeros((nrow), dtype=np.float32)
            y_imag = np.zeros((nrow), dtype=np.float32)

            xtmp = np.require(x.real, dtype=np.float32, requirements=["A", "O"])
            libspblas.SCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    csr.data.ctypes.data_as(POINTER(c_float)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    xtmp.ctypes.data_as(POINTER(c_float)), 
                    y_real.ctypes.data_as(POINTER(c_float)))

            xtmp = np.require(x.imag, dtype=np.float32, requirements=["A", "O"])
            libspblas.SCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    csr.data.ctypes.data_as(POINTER(c_float)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    xtmp.ctypes.data_as(POINTER(c_float)), 
                    y_imag.ctypes.data_as(POINTER(c_float)))
            y = y_real + 1.0j*y_imag
 
        elif csr.data.dtype == np.complex64 and x.dtype == np.float32:
            y_real = np.zeros((nrow), dtype=np.float32)
            y_imag = np.zeros((nrow), dtype=np.float32)

            datatmp = np.require(csr.data.real, dtype=np.float32, requirements=["A", "O"])
            libspblas.SCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    datatmp.ctypes.data_as(POINTER(c_float)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    x.ctypes.data_as(POINTER(c_float)), 
                    y_real.ctypes.data_as(POINTER(c_float)))

            datatmp = np.require(csr.data.imag, dtype=np.float32, requirements=["A", "O"])
            libspblas.SCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    datatmp.ctypes.data_as(POINTER(c_float)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    x.ctypes.data_as(POINTER(c_float)), 
                    y_imag.ctypes.data_as(POINTER(c_float)))
            y = y_real + 1.0j*y_imag

        elif csr.data.dtype == np.complex64 and x.dtype == np.complex64:
            raise UserWarning("Using scipy version")
            y = csr*x
            #y = np.zeros((x.shape[0]), dtype=np.complex64)
#
#            libspblas.CCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
#                    csr.data.ctypes.data_as(POINTER(c_float)),
#                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
#                    c_int(csr.indptr.shape[0]),
#                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
#                    x.ctypes.data_as(POINTER(c_float)), 
#                    y.ctypes.data_as(POINTER(c_float)))

        # Double precision
        elif csr.data.dtype == np.float64 and x.dtype == np.float64:
            y = np.zeros((nrow), dtype=np.float64)
            libspblas.DCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    csr.data.ctypes.data_as(POINTER(c_double)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    x.ctypes.data_as(POINTER(c_double)), y.ctypes.data_as(POINTER(c_double)))

        elif csr.data.dtype == np.float64 and x.dtype == np.complex128:
            y_real = np.zeros((nrow), dtype=np.float64)
            y_imag = np.zeros((nrow), dtype=np.float64)

            xtmp = np.require(x.real, dtype=np.float64, requirements=["A", "O"])
            libspblas.DCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    csr.data.ctypes.data_as(POINTER(c_double)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    xtmp.ctypes.data_as(POINTER(c_double)),
                    y_real.ctypes.data_as(POINTER(c_double)))

            xtmp = np.require(x.imag, dtype=np.float64, requirements=["A", "O"])
            libspblas.DCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    csr.data.ctypes.data_as(POINTER(c_double)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    xtmp.ctypes.data_as(POINTER(c_double)), 
                    y_imag.ctypes.data_as(POINTER(c_double)))
            y = y_real + 1.0j*y_imag
 
        elif csr.data.dtype == np.complex128 and x.dtype == np.float64:
            y_real = np.zeros((nrow), dtype=np.float64)
            y_imag = np.zeros((nrow), dtype=np.float64)

            datatmp = np.require(csr.data.real, dtype=np.float64, requirements=["A", "O"])
            libspblas.DCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    datatmp.ctypes.data_as(POINTER(c_double)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    x.ctypes.data_as(POINTER(c_double)), 
                    y_real.ctypes.data_as(POINTER(c_double)))

            datatmp = np.require(csr.data.imag, dtype=np.float64, requirements=["A", "O"])
            libspblas.DCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(nrow), 
                    datatmp.ctypes.data_as(POINTER(c_double)),
                    c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
                    c_int(csr.indptr.shape[0]),
                    csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
                    x.ctypes.data_as(POINTER(c_double)), 
                    y_imag.ctypes.data_as(POINTER(c_double)))
            y = y_real + 1.0j*y_imag

        elif csr.data.dtype == np.complex128 and x.dtype == np.complex128:
             raise UserWarning("Using scipy version")
             y = csr*x
#            y = np.zeros((x.shape[0]), dtype=np.complex128)
#
            #libspblas.ZCSRGEMV_wrapper(c_int(trans_dioc[trans]), c_int(csr.shape[0]), 
            #        csr.data.ctypes.data_as(POINTER(c_double)),
            #        c_int(csr.data.shape[0]), csr.indptr.ctypes.data_as(POINTER(c_int)), 
            #        c_int(csr.indptr.shape[0]),
            #        csr.indices.ctypes.data_as(POINTER(c_int)), c_int(csr.indices.shape[0]),
            #        x.ctypes.data_as(POINTER(c_double)), 
            #        y.ctypes.data_as(POINTER(c_double)))
        else:
            raise ValueError("Wrong dtypes")

        return y
    
    else:
        return csr*x
