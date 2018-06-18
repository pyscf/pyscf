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

def vhartree_coo(mf, dm=None, **kvargs):
  """
  Computes the matrix elements of Hartree potential
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from scipy.sparse import coo_matrix, csr_matrix
  import numpy as np

  pb,hk = mf.add_pb_hk(**kvargs)    
  dm = mf.make_rdm1() if dm is None else dm
  v_dab = pb.get_dp_vertex_sparse(sparseformat=csr_matrix)
  da2cc = pb.get_da2cc_sparse(sparseformat=csr_matrix)
  n = mf.sv.norbs
  vh_coo = coo_matrix( (v_dab.T*(da2cc*np.dot(hk, (da2cc.T*(v_dab*dm.reshape(n*n))))) ).reshape((n,n)))
  return vh_coo

