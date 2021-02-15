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
from pyscf.nao.m_overlap_ni import overlap_ni

def overlap_lil(sv, ao_log=None, funct=overlap_ni,**kvargs):
  """
  Computes the overlap matrix and returns it in List of Lists format (easy to index)
  Args:
  sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
  overlap (real-space overlap) for the whole system
  """
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from scipy.sparse import lil_matrix
  from numpy import array, int64, zeros

  aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp)
  me = aome.init_one_set(sv.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = zeros((sv.natm+1), dtype=int64)
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  sp2rcut = array([max(mu2rcut) for mu2rcut in me.ao1.sp_mu2rcut])

  n = atom2s[-1]
  lil_mat = lil_matrix((n,n))

  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      if (sp2rcut[sp1]+sp2rcut[sp2])**2<=sum((rv1-rv2)**2) : continue
      lil_mat[s1:f1,s2:f2] = funct(me,sp1,rv1,sp2,rv2,**kvargs)

  return lil_mat
