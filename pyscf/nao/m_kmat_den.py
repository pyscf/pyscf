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

def kmat_den(mf, dm=None, algo='fci', **kvargs):
  """
  Computes the matrix elements of Fock exchange operator
  Args:
    mf : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from pyscf.nao.m_prod_basis import prod_basis_c
  from scipy.sparse import csr_matrix
  import numpy as np

  pb,hk=mf.add_pb_hk(**kvargs)
  dm = mf.make_rdm1() if dm is None else dm
  algol = algo.lower()
  if algol=='fci':
    mf.fci_den = abcd2v = mf.fci_den if hasattr(mf, 'fci_den') else pb.comp_fci_den(hk)
    kmat = np.einsum('abcd,...bc->...ad', abcd2v, dm)
  else:
    print('algo=', algo)
    raise RuntimeError('unknown algoright')

  return kmat

