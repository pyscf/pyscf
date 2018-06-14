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


def init_dm_libnao(sab2dm):
  from pyscf.nao.m_libnao import libnao
  from pyscf.nao.m_sv_chain_data import sv_chain_data
  from ctypes import POINTER, c_double, c_int64
  d = np.require(sab2dm, dtype=c_double, requirements='C')
  libnao.init_sab2dm_libnao.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int64))
  libnao.init_sab2dm_libnao(d.ctypes.data_as(POINTER(c_double)), c_int64(d.shape[1]), c_int64(d.shape[0]) )
  return True
