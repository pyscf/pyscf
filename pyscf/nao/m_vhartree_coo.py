from __future__ import print_function, division

def vhartree_coo(sv, dm=None, **kvargs):
  """
  Computes the matrix elements of Hartree potential
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from pyscf.nao.m_prod_basis import prod_basis_c
  from scipy.sparse import coo_matrix
  import numpy as np

  if hasattr(sv, 'pb'):
    pb = sv.pb
  else:
    pb = sv.pb = prod_basis_c().init_prod_basis_pp(sv, **kvargs)
    sv.hkernel_den = pb.comp_coulomb_den()
    
  dm = sv.comp_dm() if dm is None else dm
  v_dab = pb.get_dp_vertex_sparse()
  da2cc = pb.get_da2cc_sparse()
  
  n = sv.norbs
  vh_coo = coo_matrix( (v_dab.T*(da2cc*np.dot(sv.hkernel_den, (da2cc.T*(v_dab*dm.reshape(n*n))))) ).reshape((n,n)))
  return vh_coo

