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

def vnucele_coo_subtract(sv, **kvargs):
  """
  Computes the matrix elements defined by 
    Vne = H_KS - T - V_H - V_xc
  which serve as nuclear-electron attraction matrix elements for pseudo-potential DFT calculations
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  tkin = 0.5*(sv.laplace_coo().tocsr())
  vhar = sv.vhartree_coo(**kvargs).tocsr()
  vxc  = sv.vxc_lil(**kvargs).tocsr()
  vne =  sv.get_hamiltonian(**kvargs)[0].tocsr()-tkin-vhar-vxc
  return vne.tocoo()

