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

def vnucele_coo(sv, algo=None, **kvargs):
  """
  Computes the nucleus-electron attraction matrix elements
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements

      These are tricky to define. In case of all-electron calculations it is well known, but 
      in case of pseudo-potential calculations we need some a specification of pseudo-potential
      and a specification of (Kleinman-Bulander) projectors to compute explicitly. Practically,
      we will subtract the computed matrix elements from the total Hamiltonian to find out the 
      nuclear-electron interaction in case of SIESTA import. This means that Vne is defined by 
        Vne = H_KS - T - V_H - V_xc

  """
  if algo is None: 
    if hasattr(sv, 'xml_dict'): 
      vne_coo = sv.vnucele_coo_subtract(**kvargs) # try to subtract if data is coming from SIESTA
    else:
      vne_coo = sv.vnucele_coo_coulomb(**kvargs)  # try to compute the Coulomb attraction matrix elements
  else:
    vne_coo = sv.algo(**kvargs)

  return vne_coo

