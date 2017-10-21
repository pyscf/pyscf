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

    def __init__(self, GPU, x, v_dab, ksn2f, ksn2e, cc_da, norbs, nfermi, nprod, vstart):
        
        if GPU and GPU_import:
            self.GPU=True

            self.norbs = norbs
            self.nfermi = nfermi
            self.nprod = nprod
            self.vstart = vstart
            self.nvirt = norbs-vstart
            self.v_dab = v_dab
            self.cc_da = cc_da
            self.x = x[0, 0, :, :, 0]
            
            self.block_size = np.array([32, 32], dtype=np.int32) # threads by block
            self.grid_size = np.array([0, 0], dtype=np.int32) # number of blocks
            dimensions = [self.nfermi, ksn2f.shape[2]]

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

        vext_real = np.copy(v.real)
        vext_imag = np.copy(v.imag)

        
        nm2v_re = np.zeros((self.nfermi*self.nvirt), dtype=np.float32)
        nm2v_im = np.zeros((self.nfermi*self.nvirt), dtype=np.float32)

        print("Aqui??", self.nvirt, self.vstart)
        libnao_gpu.apply_rf0_device(vext_real.ctypes.data_as(POINTER(c_float)),
                vext_imag.ctypes.data_as(POINTER(c_float)),
                nm2v_re.ctypes.data_as(POINTER(c_float)), 
                nm2v_im.ctypes.data_as(POINTER(c_float)))


        # ref real part
        vdp = csr_matvec(self.cc_da, vext_real)
        sab_ref = (vdp*self.v_dab)#.reshape([no,no])
        
        xocc = self.x[0:self.nfermi, :]
        xvrt = self.x[self.vstart:, :]
        nb2v_ref = blas.sgemm(1.0, xocc, sab_ref.reshape([self.norbs, self.norbs]))
        nm2v_re_ref = blas.sgemm(1.0, nb2v_ref, xvrt.T)

        # imag
        vdp = csr_matvec(self.cc_da, vext_imag)
        sab_ref = (vdp*self.v_dab)#.reshape([no,no])
        
        xocc = self.x[0:self.nfermi, :]
        xvrt = self.x[self.vstart:, :]
        nb2v_ref = blas.sgemm(1.0, xocc, sab_ref.reshape([self.norbs, self.norbs]))
        nm2v_im_ref = blas.sgemm(1.0, nb2v_ref, xvrt.T)
        
        print("check real: ", np.sum(abs(nm2v_re)), np.sum(abs(nm2v_re_ref)), "Error: ", np.sum(abs(nm2v_re.reshape([self.nfermi, self.nvirt]) - nm2v_re_ref)))
        print("check imag: ", np.sum(abs(nm2v_im)), np.sum(abs(nm2v_im_ref)), "Error: ", np.sum(abs(nm2v_im.reshape([self.nfermi, self.nvirt]) - nm2v_im_ref)))
        

        import sys
        sys.exit()

        return vdp

    def clean_gpu(self):
        libnao_gpu.free_device()
