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
        """
          Input Parameters:
          -----------------
            GPU: variable to set up GPU calculations. It can take several forms
                * None : if GPU=None, no GPU will be use
                * True : if GPU is True, then calculation will be using GPUs with 
                      default setup
                * False : Identic to None
                * dictionary: a dictionary containing the different parameters
                    for the gpu setup, the keys are,
                      * use, booleean to know if wew will use GPU calculations
                      * device: integer to use a certain GPU if there is more than one
        """
        
        if (isinstance(GPU, dict) or GPU == True) and GPU_import:

            list_kw = ["use", "device", "gpu count"]
            default = [True, 0, self.countGPUs()]
            self.GPU = dict()
            
            if isinstance(GPU, dict):
              for key, val in zip(list_kw, default):
                if key in GPU.keys():
                  self.GPU[key] = GPU[key]
                else:
                  self.GPU[key] = val 
            elif GPU: # GPU is True
              for key, val in zip(list_kw, default):
                self.GPU[key] = val
            else:
              raise ValueError("wrong input for GPU")

            if not self.GPU["use"]:
              self.GPU = None # lets keep None for the next
            else:

              print(self.GPU)
              if isinstance(self.GPU["device"], int):
                if self.GPU["device"] < self.GPU["gpu count"] and \
                    self.GPU["device"] >= 0:
                  self.setDevice(self.GPU["device"])
                else:
                  mess = """
                          GPU['device'] = {0}
                          but there is only {1} gpus on this system.
                         """.format(self.GPU["device"], self.GPU["gpu count"])
                  raise ValueError(mess)
              else:
                raise ValueError("GPU['device'] must be an integer, no multi GPU support at the moment.")
              
              self.norbs = norbs
              self.nfermi = nfermi[0] # taking only the first spin, will need to be corrected
              self.nprod = nprod
              self.vstart = vstart[0]
              self.nvirt = self.norbs-self.vstart
              
              self.block_size = np.array([32, 32], dtype=np.int32) # threads by block
              self.grid_size = np.array([0, 0], dtype=np.int32) # number of blocks
              dimensions = [nfermi, nprod]

              for i in range(2):
                  if dimensions[i] <= self.block_size[i]:
                      self.block_size[i] = dimensions[i]
                      self.grid_size[i] = 1
                  else:
                      self.grid_size[i] = dimensions[i]/self.block_size[i] + 1

              #print(X4.shape)
              #print(ksn2e.shape)
              #print(ksn2f.shape)
              #print(self.nfermi, self.norbs, self.nprod, self.vstart, self.nvirt)
              libnao_gpu.init_tddft_iter_gpu(
                          X4.ctypes.data_as(POINTER(c_float)), c_int(self.norbs),
                          ksn2e[0, 0, :].ctypes.data_as(POINTER(c_float)), 
                          ksn2f[0, 0, :].ctypes.data_as(POINTER(c_float)),
                          c_int(self.nfermi), c_int(self.nprod), c_int(self.vstart))

        elif (isinstance(GPU, dict) or GPU == True) and not GPU_import:
            if isinstance(GPU, dict):
                if GPU["use"]:
                    raise ValueError("GPU lib failed to initialize!")
                else:
                    self.GPU = None
            else:
                raise ValueError("GPU lib failed to initialize!")
        else:
            self.GPU = None

    def countGPUs(self):
      """
        Return the number of devices available for the calculations
      """
      return libnao_gpu.CountDevices()

    def setDevice(self, gpu_id):
      
      libnao_gpu.SetDevice(c_int(gpu_id))

    def getDevice(self):
      
      return libnao_gpu.GetDevice()


    def cpy_sab_to_device(self, sab, Async=-1):
        """
            Async can take the following values:
                * 0 default stream
                * 1 real stream
                * 2 imag stream
                * -1 or any other value: Not using Async, just blocking memcpy
        """

        libnao_gpu.memcpy_sab_host2device(sab.ctypes.data_as(POINTER(c_float)), c_int(Async))

    def cpy_sab_to_host(self, sab, Async=-1):
        """
            Async can take the following values:
                * 0 default stream
                * 1 real stream
                * 2 imag stream
                * -1 or any other value: Not using Async, just blocking memcpy
        """
 
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
