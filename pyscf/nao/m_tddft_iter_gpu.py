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

from __future__ import print_function, division
import numpy as np
from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int
import sys

try:
    # try import gpu library
    from pyscf.nao.m_libnao import libnao_gpu
    GPU_import = True
except:
    GPU_import = False


class tddft_iter_gpu_c():

    def __init__(self, GPU, v_dab, ksn2f, ksn2e, norbs, nfermi, vstart):
        
        if GPU and GPU_import:
            self.GPU=True

            self.norbs = norbs
            self.nfermi = nfermi
            self.vstart = vstart
            self.v_dab = v_dab
            
            self.block_size = np.array([32, 32], dtype=np.int32) # threads by block
            self.grid_size = np.array([0, 0], dtype=np.int32) # number of blocks
            dimensions = [self.nfermi, ksn2f.shape[2]]
            for i in range(2):
                if dimensions[i] <= self.block_size[i]:
                    self.block_size[i] = dimensions[i]
                    self.grid_size[i] = 1
                else:
                    self.grid_size[i] = dimensions[i]/self.block_size[i] + 1
            libnao_gpu.init_iter_gpu(x[0, 0, :, :, 0].ctypes.data_as(POINTER(c_float)), c_int64(self.norbs),
                ksn2e[0, 0, :].ctypes.data_as(POINTER(c_float)), c_int64(ksn2e[0, 0, :].size),
                ksn2f[0, 0, :].ctypes.data_as(POINTER(c_float)), c_int64(ksn2f[0, 0, :].size),
                c_int64(self.nfermi), c_int64(self.vstart))
        elif GPU and not GPU_import:
            raise ValueError("GPU lib failed to initialize!")
        else:
            self.GPU = False

    def apply_rf0_gpu(self, xocc, sab, comega):

        nb2v = xocc*sab.real
        libnao_gpu.calc_nm2v_real(nb2v.ctypes.data_as(POINTER(c_float)))
        
        nb2v = xocc*sab.imag
        libnao_gpu.calc_nm2v_imag(nb2v.ctypes.data_as(POINTER(c_float)))

        libnao_gpu.calc_XXVV(c_double(comega.real), c_double(comega.imag),
                self.block_size.ctypes.data_as(POINTER(c_int)), self.grid_size.ctypes.data_as(POINTER(c_int)))

        ab2v = np.zeros([self.norbs*self.norbs], dtype=np.float32)
        
        libnao_gpu.calc_ab2v_imag(ab2v.ctypes.data_as(POINTER(c_float)))
        vdp = 1j*self.v_dab*ab2v

        libnao_gpu.calc_ab2v_real(ab2v.ctypes.data_as(POINTER(c_float)))
        vdp += self.v_dab*ab2v

        return vdp

    def clean_gpu(self):
        libnao_gpu.clean_gpu()
