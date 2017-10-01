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
  from scipy.sparse import csr_matrix
  import numpy as np

  pb,hk = sv.add_pb_hk(**kvargs)    
  dm = sv.comp_dm() if dm is None else dm
  v_dab = pb.get_dp_vertex_sparse(sparseformat=csr_matrix)
  da2cc = pb.get_da2cc_sparse(sparseformat=csr_matrix)
  n = sv.norbs
  vh_coo = coo_matrix( (v_dab.T*(da2cc*np.dot(hk, (da2cc.T*(v_dab*dm.reshape(n*n))))) ).reshape((n,n)))
  return vh_coo

