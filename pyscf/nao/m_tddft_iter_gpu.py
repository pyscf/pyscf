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

    def __init__(self, GPU, X4, ksn2f, ksn2e, norbs, nfermi, nprod, vstart):
        
        if GPU and GPU_import:
            self.GPU=True

            self.norbs = norbs
            self.nfermi = nfermi
            self.nprod = nprod
            self.vstart = vstart
            self.nvirt = norbs-vstart
            
            self.block_size = np.array([32, 32], dtype=np.int32) # threads by block
            self.grid_size = np.array([0, 0], dtype=np.int32) # number of blocks
            dimensions = [nfermi, nprod]

            for i in range(2):
                if dimensions[i] <= self.block_size[i]:
                    self.block_size[i] = dimensions[i]
                    self.grid_size[i] = 1
                else:
                    self.grid_size[i] = dimensions[i]/self.block_size[i] + 1

            libnao_gpu.init_tddft_iter_gpu(
                        X4.ctypes.data_as(POINTER(c_float)), c_int(norbs),
                        ksn2e.ctypes.data_as(POINTER(c_float)), 
                        ksn2f.ctypes.data_as(POINTER(c_float)),
                        c_int(nfermi), c_int(nprod), c_int(vstart))

        elif GPU and not GPU_import:
            raise ValueError("GPU lib failed to initialize!")
        else:
            self.GPU = False

    def cpy_sab_to_device(self, sab, Async=0):
        libnao_gpu.memcpy_sab_host2device(sab.ctypes.data_as(POINTER(c_float)), c_int(Async))

    def cpy_sab_to_host(self, sab, Async=0):
        libnao_gpu.memcpy_sab_device2host(sab.ctypes.data_as(POINTER(c_float)), c_int(Async))

    def calc_nb2v_from_sab(self, reim):
        libnao_gpu.calc_nb2v_from_sab(c_int(reim))

    def calc_nm2v_real(self):
        libnao_gpu.get_nm2v_real()
    
    def calc_nm2v_imag(self):
        libnao_gpu.get_nm2v_imag()

    def calc_nb2v_from_nm2v_real(self):
        libnao_gpu.calc_nb2v_from_nm2v_real()

    def calc_nb2v_from_nm2v_imag(self):
        libnao_gpu.calc_nb2v_from_nm2v_imag()

    def calc_sab(self, reim):
        libnao_gpu.get_sab(c_int(reim))

    def div_eigenenergy_gpu(self, comega):
        libnao_gpu.div_eigenenergy_gpu(c_double(comega.real), c_double(comega.imag),
                self.block_size.ctypes.data_as(POINTER(c_int)),
                self.grid_size.ctypes.data_as(POINTER(c_int)))

    def clean_gpu(self):
        libnao_gpu.free_device()
