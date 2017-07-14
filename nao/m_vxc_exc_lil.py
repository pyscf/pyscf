from __future__ import print_function, division

#
#
#
def vxc_exc_lil(sv, sab2dm, ao_log=None, **kvargs):
  """
    Computes the exchange-correlation matrix elements
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
    Returns:
      vxc,exc
  """
  from pyscf.nao.m_xc_scalar_ni import xc_scalar_ni as funct
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from scipy.sparse import lil_matrix
  from numpy import array, int64, zeros

  aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp, sv, sab2dm)
  me = aome.init_one_set(sv.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = zeros((sv.natm+1), dtype=int64)
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  sp2rcut = array([max(mu2rcut) for mu2rcut in me.ao1.sp_mu2rcut])
  
  lil = lil_matrix((atom2s[-1],atom2s[-1]))
  
  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      if (sp2rcut[sp1]+sp2rcut[sp2])**2<=sum((rv1-rv2)**2) : continue
      lil[s1:f1,s2:f2] = funct(me,sp1,rv1,sp2,rv2,**kvargs)

  return lil
