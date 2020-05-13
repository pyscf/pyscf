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

#
def dm(ksnac2x, ksn2occ):
  """
    Computes the density matrix 
    Args:
      ksnar2x : eigenvectors
      ksn2occ : occupations
    Returns:
      ksabc2dm : 
  """
  from numpy import einsum, zeros_like

  ksnac2x_occ = einsum('ksnac,ksn->ksnac', ksnac2x, ksn2occ)
  ksabc2dm = einsum('ksnac,ksnbc->ksabc', ksnac2x_occ, ksnac2x)

  print(ksabc2dm.shape)

  return ksabc2dm
