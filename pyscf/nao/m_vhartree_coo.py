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

def dm2j_fullmat(n, v_dab, da2cc, hk, dm):
  return (v_dab.T*(da2cc*np.dot(hk, (da2cc.T*(v_dab*dm.reshape(n*n))))) ).reshape((n,n))

def vhartree_coo(mf, dm=None, **kw):
  """
  Computes the matrix elements of Hartree potential
  Args:
    mf: this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from scipy.sparse import coo_matrix, csr_matrix

  pb,hk = mf.add_pb_hk(**kw)
  dm = mf.make_rdm1() if dm is None else dm
  v_dab = pb.get_dp_vertex_sparse(sparseformat=csr_matrix)
  da2cc = pb.get_da2cc_sparse(sparseformat=csr_matrix)

  n = mf.norbs
  nspin = mf.nspin
  if mf.nspin==1:
    dm = dm.reshape((n,n))
    vh_coo = coo_matrix( dm2j_fullmat(n, v_dab, da2cc, hk, dm) )
  elif mf.nspin==2:
    dm = dm.reshape((nspin,n,n))
    vh_coo = [coo_matrix( dm2j_fullmat(n, v_dab, da2cc, hk, dm[0,:,:]) ),
      coo_matrix( dm2j_fullmat(n, v_dab, da2cc, hk, dm[1,:,:]) )]
  else:
    print(nspin)
    raise RuntimeError('nspin>2?')

  return vh_coo
