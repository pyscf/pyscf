from __future__ import print_function, division
import numpy as np

#
#
#
def comp_overlap_coo(sv):
  """
    Computes the overlap matrix and returns it in coo format (simplest sparse format to construct)
    Args:
      sv : (System Variables), this must have arrays of coordinates and ao_log class
    Returns:
      overlap
  """
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from scipy.sparse import coo_matrix
  sp2rcut = np.array([max(mu2rcut) for mu2rcut in sv.sp_mu2rcut])
  me = ao_matelem_c(sv.ao_log)

  nnz = 0
  for sp1,rv1 in zip(sv.atom2sp,sv.atom2coord):
    for sp2,rv2 in zip(sv.atom2sp,sv.atom2coord):
      if (sp2rcut[sp1]+sp2rcut[sp2])**2>sum((rv1-rv2)**2) : nnz = nnz + me.sp2norbs[sp1]*me.sp2norbs[sp2]
  
  irow,icol,data = np.zeros((nnz), dtype='int64'),np.zeros((nnz), dtype='int64'),np.zeros((nnz), dtype='float64') # Start to construct coo matrix

  atom2s = np.zeros((sv.natoms+1), dtype='int32')
  for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+me.sp2norbs[sp]
  
  inz=-1
  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(sv.atom2sp,sv.atom2coord,atom2s,atom2s[1:])):
      if (sp2rcut[sp1]+sp2rcut[sp2])**2<=sum((rv1-rv2)**2) : continue
      oo = me.get_overlap_ap(sp1,sp2,rv1,rv2)
      for o1 in range(s1,f1):
        for o2 in range(s2,f2):
          inz = inz+1
          irow[inz],icol[inz],data[inz] = o1,o2,oo[o1-s1,o2-s2]

  return coo_matrix((data, (irow, icol)), shape=(sv.norbs, sv.norbs))
