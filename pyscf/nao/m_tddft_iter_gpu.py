from __future__ import print_function, division
import numpy as np
from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int, c_long
from scipy.linalg import blas
from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec, csc_matvecs
import sys

try: # to import gpu library
  from pyscf.lib import misc
  libnao_gpu = misc.load_library("libnao_gpu")
  GPU_import = True
except:
  GPU_import = False

class tddft_iter_gpu_c():

    def __init__(self, GPU, x, v_dab, ksn2f, ksn2e, cc_da, moms0, norbs, nfermi, nprod, vstart):
        self.rf0_ncalls = 0
        self.l0_ncalls = 0
        
        if GPU and GPU_import:
            self.GPU=True

            self.norbs = norbs
            self.nfermi = nfermi
            self.nprod = nprod
            self.vstart = vstart
            self.nvirt = norbs-vstart
            self.x = x[0, 0, :, :, 0]
            self.moms0 = moms0
            
            self.block_size = np.array([32, 32], dtype=np.int32) # threads by block
            self.grid_size = np.array([0, 0], dtype=np.int32) # number of blocks
            dimensions = [nfermi, nprod]

            for i in range(2):
                if dimensions[i] <= self.block_size[i]:
                    self.block_size[i] = dimensions[i]
                    self.grid_size[i] = 1
                else:
                    self.grid_size[i] = dimensions[i]/self.block_size[i] + 1

            libnao_gpu.init_tddft_iter_gpu(self.x.ctypes.data_as(POINTER(c_float)), c_int(self.norbs),
                ksn2e[0, 0, :].ctypes.data_as(POINTER(c_float)), ksn2f[0, 0, :].ctypes.data_as(POINTER(c_float)),
                c_int(self.nfermi), c_int(self.nprod), c_int(self.vstart), 
                cc_da.data.ctypes.data_as(POINTER(c_float)), cc_da.indptr.ctypes.data_as(POINTER(c_int)),
                cc_da.indices.ctypes.data_as(POINTER(c_int)), np.array(cc_da.shape, dtype=np.int32).ctypes.data_as(POINTER(c_int)),
                c_int(cc_da.nnz), c_int(cc_da.indptr.size),
                v_dab.data.ctypes.data_as(POINTER(c_float)), v_dab.indptr.ctypes.data_as(POINTER(c_int)),
                v_dab.indices.ctypes.data_as(POINTER(c_int)), np.array(v_dab.shape, dtype=np.int32).ctypes.data_as(POINTER(c_int)),
                c_int(v_dab.nnz), c_int(v_dab.indptr.size))

        elif GPU and not GPU_import:
            raise ValueError("GPU lib failed to initialize!")
        else:
            self.GPU = False

    def apply_rf0_gpu(self, v, comega=1j*0.0):
        assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
        self.rf0_ncalls+=1

        vext_real = np.copy(v.real)
        vext_imag = np.copy(v.imag)

        chi0_re = np.zeros((self.nprod), dtype=np.float32)
        chi0_im = np.zeros((self.nprod), dtype=np.float32)

        libnao_gpu.apply_rf0_device(vext_real.ctypes.data_as(POINTER(c_float)),
                vext_imag.ctypes.data_as(POINTER(c_float)),
                c_double(comega.real), c_double(comega.imag),
                chi0_re.ctypes.data_as(POINTER(c_float)),
                chi0_im.ctypes.data_as(POINTER(c_float)),
                self.block_size.ctypes.data_as(POINTER(c_int)), 
                self.grid_size.ctypes.data_as(POINTER(c_int)))

        return chi0_re + 1.0j*chi0_im

    def clean_gpu(self):
        libnao_gpu.free_device()
