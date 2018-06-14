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
from pyscf.nao.m_dipole_ni import dipole_ni

#
#
#
def dipole_coo(sv, ao_log=None, funct=dipole_ni, **kvargs):
  """
    Computes the dipole matrix and returns it in coo format (simplest sparse format to construct)
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
    Returns:
      overlap (real-space overlap) for the whole system
  """
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from scipy.sparse import coo_matrix
  from numpy import array, int64, zeros

  aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp)
  me = aome.init_one_set(sv.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = zeros((sv.natm+1), dtype=int64)
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  sp2rcut = array([max(mu2rcut) for mu2rcut in me.ao1.sp_mu2rcut])

  nnz = 0
  for sp1,rv1 in zip(sv.atom2sp,sv.atom2coord):
    n1,rc1 = me.ao1.sp2norbs[sp1],sp2rcut[sp1]
    for sp2,rv2 in zip(sv.atom2sp,sv.atom2coord):
      if (rc1+sp2rcut[sp2])**2>((rv1-rv2)**2).sum() : nnz = nnz + n1*me.ao1.sp2norbs[sp2]

  irow,icol,data = zeros(nnz, dtype=int64),zeros(nnz, dtype=int64),zeros((3,nnz)) # Start to construct three coo matrices
  
  inz=-1
  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      if (sp2rcut[sp1]+sp2rcut[sp2])**2<=sum((rv1-rv2)**2) : continue
      dd = funct(me,sp1,rv1,sp2,rv2,**kvargs)
      for o1 in range(s1,f1):
        for o2 in range(s2,f2):
          inz = inz+1
          irow[inz],icol[inz],data[:,inz] = o1,o2,dd[:,o1-s1,o2-s2]

  norbs = atom2s[-1]
  sh = (norbs,norbs)
  rc = (irow,icol)
  return coo_matrix((data[0], rc), shape=sh),coo_matrix((data[1], rc), shape=sh),coo_matrix((data[2],rc), shape=sh)
