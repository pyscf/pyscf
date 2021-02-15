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
from pyscf.nao.m_coulomb_am import coulomb_am
import numpy as np

#
#
#
def comp_coulomb_pack(sv, ao_log=None, funct=coulomb_am, dtype=np.float64, **kw):
  """
    Computes the matrix elements given by funct, for instance coulomb interaction
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
      ao_log : description of functions (either orbitals or product basis functions)
    Returns:
      matrix elements for the whole system in packed form (lower triangular part)
  """
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from pyscf.nao.m_pack2den import cp_block_pack_u
  
  aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp)
  me = ao_matelem_c(sv.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = np.zeros((sv.natm+1), dtype=np.int64)
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  norbs = atom2s[-1]

  res = np.zeros(norbs*(norbs+1)//2, dtype=dtype)

  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    #print("atom1 = {0}, rv1 = {1}".format(atom1, rv1))
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      if atom2>atom1: continue # skip 
      oo2f = funct(me,sp1,rv1,sp2,rv2, **kw)
      cp_block_pack_u(oo2f, s1,f1,s2,f2, res)

  return res, norbs
