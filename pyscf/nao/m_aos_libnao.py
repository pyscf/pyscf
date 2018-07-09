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

from __future__ import print_function
import numpy as np
from ctypes import POINTER, c_double, c_int64
from pyscf.nao.m_libnao import libnao

libnao.aos_libnao.argtypes = (
  POINTER(c_int64),  # ncoords
  POINTER(c_double), # coords
  POINTER(c_int64),  # norbs
  POINTER(c_double), # res[icoord, orb]
  POINTER(c_int64))  # ldres leading dimension (fastest changing dimension) of res (norbs)


""" The purpose of this is to evaluate the atomic orbitals at a given set of atomic coordinates """
def aos_libnao(coords, norbs):
  assert len(coords.shape) == 2
  assert coords.shape[1] == 3
  assert norbs>0

  ncoords = coords.shape[0]
  co2val = np.require( np.zeros((ncoords,norbs)), dtype=c_double, requirements='CW')
  crd_copy = np.require(coords, dtype=c_double, requirements='C')

  libnao.aos_libnao(
    c_int64(ncoords),
    crd_copy.ctypes.data_as(POINTER(c_double)),
    c_int64(norbs),
    co2val.ctypes.data_as(POINTER(c_double)),
    c_int64(norbs))

  return co2val




