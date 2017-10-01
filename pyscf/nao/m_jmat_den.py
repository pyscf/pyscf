from __future__ import print_function, division

def jmat_den(sv, dm=None, **kvargs):
  """
  Computes the matrix elements of Fock operator
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from pyscf.nao.m_prod_basis import prod_basis_c
  from scipy.sparse import csr_matrix
  import numpy as np

  pb,hk=sv.add_pb_hk(**kvargs)    
  dm = sv.comp_dm() if dm is None else dm  
  n = sv.norbs
  da2cc = pb.get_da2cc_sparse(sparseformat=csr_matrix)
  jmat = zeros((n,n))
  return jmat

