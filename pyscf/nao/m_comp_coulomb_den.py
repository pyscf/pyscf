from __future__ import print_function, division
from pyscf.nao.m_coulomb_am import coulomb_am
import numpy as np

#
#
#
def comp_coulomb_den(sv, ao_log=None, funct=coulomb_am, dtype=np.float64, **kvargs):
  """
    Computes the matrix elements given by funct, for instance coulomb interaction
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
    Returns:
      matrix elements (real-space overlap) for the whole system
  """
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  
  aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp)
  me = aome.init_one_set(sv.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = np.zeros((sv.natm+1), dtype=np.int32)
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  norbs = atom2s[-1]

  res = np.zeros((norbs,norbs), dtype=dtype)

  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      oo2f = funct(me,sp1,rv1,sp2,rv2,**kvargs)
      res[s1:f1,s2:f2] = oo2f[:,:]

  return res

#
#
#
if __name__=='__main__':
  from pyscf.nao.m_comp_coulomb_den import comp_coulomb_den
  from pyscf.nao.m_comp_overlap_coo import comp_overlap_coo
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_overlap_am import overlap_am

  sv = system_vars_c(label='siesta')

  over_coo = comp_overlap_coo(sv, funct=overlap_am).toarray()
  over_den = comp_coulomb_den(sv, funct=overlap_am)
  
  print(np.allclose(over_coo, over_den)) # must be always true
  
