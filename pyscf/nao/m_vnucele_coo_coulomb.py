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

def vnucele_coo_coulomb(sv, **kvargs):
  """
  Computes the matrix elements defined by 
    Vne = f(r) sum_a   Z_a/|r-R_a|  g(r)
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from numpy import einsum, dot
  from scipy.sparse import coo_matrix
  g = sv.build_3dgrid_ae(**kvargs)
  ca2o = sv.comp_aos_den(g.coords)
  vnuc = sv.comp_vnuc_coulomb(g.coords)
  vnuc_w = g.weights*vnuc
  cb2vo = einsum('co,c->co', ca2o, vnuc_w)
  vne = dot(ca2o.T,cb2vo)
  return coo_matrix(vne)

