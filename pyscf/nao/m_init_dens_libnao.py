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
from ctypes import POINTER, c_int64, byref

libnao.init_dens_libnao.argtypes = ( POINTER(c_int64), ) # info

def init_dens_libnao():
  """ Initilize the auxiliary for computing the density on libnao site """

  info = c_int64(-999)
  libnao.init_dens_libnao( byref(info))

  return info.value
