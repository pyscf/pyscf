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

from __future__ import division
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import pycuda.autoinit

kernel_code_div_eigenenergy_cuda = """
#include<stdio.h>
#include<stdlib.h>

__global__ void calc_XXVV_gpu(float *nm2v_re, float *nm2v_im, int nm2v_dim1, int nm2v_dim2,
    float *ksn2e, float *ksn2f, int nfermi, int vstart, int ksn2e_dim, double omega_re,
    double omega_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //nocc
    int j = blockIdx.y * blockDim.y + threadIdx.y; //nvirt
    int m, index;
    float en, em, fn, fm;
    double alpha, beta, a, b;

    if (i < nfermi)
    {
        en = ksn2e[i];
        fn = ksn2f[i];
        if ( (j < ksn2e_dim - i -1))
        {
            m = j + i + 1 - vstart;
            if (m > 0)
            {
                em = ksn2e[i+1+j];
                fm = ksn2f[i+1+j];
                a = (omega_re - (em-en))*(omega_re - (em-en)) + omega_im*omega_im;
                b = (omega_re + (em-en))*(omega_re + (em-en)) + omega_im*omega_im;

                alpha =  (b*(omega_re - (em-en)) - a*(omega_re + (em-en)))/(a*b);
                beta = omega_im*(a-b)/(a*b);

                index = i*nm2v_dim2 + m;
                nm2v_re[index] = (fn - fm) * (nm2v_re[index]*alpha - nm2v_im[index]*beta);
                nm2v_im[index] = (fn - fm) * (nm2v_re[index]*beta + nm2v_im[index]*alpha);
            }
        }
    }
}
"""

def div_eigenenergy_cuda(ksn2e, ksn2f, nfermi, vstart, comega, nm2v_re, nm2v_im,
        block_size, grid_size):

    block = (int(block_size[0]), int(block_size[1]), int(1))
    grid = (int(grid_size[0]), int(grid_size[1]))

    mod = SourceModule(kernel_code_div_eigenenergy_cuda)
    calc_XXVV = mod.get_function("calc_XXVV_gpu")
    calc_XXVV(nm2v_re, nm2v_im, np.int32(nm2v_re.shape[0]),
        np.int32(nm2v_re.shape[1]), ksn2e, ksn2f, np.int32(nfermi),
        np.int32(vstart), np.int32(ksn2e.shape[0]), np.float64(comega.real),
        np.float64(comega.imag), block = block, grid = grid)
