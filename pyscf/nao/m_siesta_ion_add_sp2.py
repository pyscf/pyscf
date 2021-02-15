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

import numpy as np
from pyscf.nao.m_get_sp_mu2s import get_sp_mu2s

#
#
#
def _siesta_ion_add_sp2(self, sp2ion):
  """
    Adds fields sp2nmult, sp_mu2rcut and sp_mu2j  to the gived object self, based on sp2ion
  """
  self.nspecies = len(sp2ion)
  if self.nspecies<1: return

  self.sp2nmult = np.array([ion["paos"]["npaos"] for ion in sp2ion], dtype='int64')

  self.sp_mu2rcut = [np.array(ion["paos"]["cutoff"], dtype='float64') for ion in sp2ion]

  self.sp_mu2j = []
  for sp,ion in enumerate(sp2ion):
    npao = len(ion["paos"]["orbital"])
    self.sp_mu2j.append(np.array([ion["paos"]["orbital"][mu]['l'] for mu in range(npao)], dtype='int64'))
  
  self.sp_mu2s = get_sp_mu2s(self.sp2nmult, self.sp_mu2j)
