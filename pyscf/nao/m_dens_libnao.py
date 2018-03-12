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
from pyscf.nao.m_libnao import libnao
from ctypes import POINTER, c_double, c_int64
from numpy import require, zeros

libnao.dens_libnao.argtypes = (
  POINTER(c_double),# crds[i,0:3] 
  POINTER(c_int64), # ncrds
  POINTER(c_double),# dens[i,0:]  
  POINTER(c_int64)) # ndens


def dens_libnao(crds, nspin):
  """  Compute the electronic density using library call """
  assert crds.ndim==2  
  assert crds.shape[-1]==3
  
  nc = crds.shape[0]
  crds_cp = require(crds, dtype=c_double, requirements='C')
  dens = require( zeros((nc, nspin)), dtype=c_double, requirements='CW')
  
  libnao.dens_libnao(
    crds_cp.ctypes.data_as(POINTER(c_double)),
    c_int64(nc),
    dens.ctypes.data_as(POINTER(c_double)),
    c_int64(nspin))

  return dens
