from __future__ import print_function, division
from pyscf.nao.m_coulomb_am import coulomb_am
import numpy as np

#
#
#
def comp_coulomb_pack(sv, ao_log=None, funct=coulomb_am, dtype=np.float64, **kvargs):
  """
    Computes the matrix elements given by funct, for instance coulomb interaction
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
      ao_log : description of functions (either orbitals or product basis functions)
    Returns:
      matrix elements (real-space overlap) for the whole system
  """
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from pyscf.nao.m_pack2den import triu_indices 
  
  aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp)
  me = ao_matelem_c(sv.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = np.zeros((sv.natm+1), dtype=np.int64)
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  norbs = atom2s[-1]

  res = np.zeros(norbs*(norbs+1)//2, dtype=dtype)
  ind = triu_indices(norbs)

  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      #if atom2>atom1: continue # skip 
      oo2f = funct(me,sp1,rv1,sp2,rv2,**kvargs)
      for i1 in range(s1,f1):
        for i2 in range(s2,f2):
          if ind[i1, i2] >= 0:
              res[ind[i1, i2]] = oo2f[i1-s1,i2-s2]

  return res, norbs
