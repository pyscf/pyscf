from __future__ import print_function, division
from numpy import array, int64, zeros, float64
try:
    import numba as nb
    from pyscf.nao.m_numba_utils import fill_triu
    use_numba = True
except:
    use_numba = False


#
#
#
def vxc_pack(sv, dm, xc_code, deriv, ao_log=None, dtype=float64, **kvargs):
  """
    Computes the exchange-correlation matrix elements packed version (upper triangular)
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
    Returns:
      vxc,exc
  """
  from pyscf.nao.m_xc_scalar_ni import xc_scalar_ni
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from pyscf.nao.m_pack2den import triu_indices

  aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp, sv, dm)
  me = aome.init_one_set(sv.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = zeros((sv.natm+1), dtype=int64)
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  sp2rcut = array([max(mu2rcut) for mu2rcut in me.ao1.sp_mu2rcut])
  norbs = atom2s[-1]
  
  lil = zeros(norbs*(norbs+1)//2, dtype=dtype)
  ind = triu_indices(norbs)

  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      if (sp2rcut[sp1]+sp2rcut[sp2])**2<=sum((rv1-rv2)**2) : continue
      xc = xc_scalar_ni(me,sp1,rv1,sp2,rv2,xc_code,deriv,**kvargs)
      if use_numba:
          fill_triu(xc, ind, lil, s1, f1, s2, f2)
      else:
          for i1 in range(s1,f1):
             for i2 in range(s2,f2):
               if ind[i1, i2] >= 0:
                   lil[ind[i1, i2]] = xc[i1-s1,i2-s2]

  return lil
