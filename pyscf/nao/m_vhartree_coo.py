from __future__ import print_function, division

def vhartree_coo(mf, dm=None, **kvargs):
  """
  Computes the matrix elements of Hartree potential
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from scipy.sparse import coo_matrix, csr_matrix
  import numpy as np

  pb,hk = mf.add_pb_hk(**kvargs)    
  dm = mf.make_rdm1() if dm is None else dm
  v_dab = pb.get_dp_vertex_sparse(sparseformat=csr_matrix)
  da2cc = pb.get_da2cc_sparse(sparseformat=csr_matrix)
  n = dm.shape[-2]
  vh_coo = coo_matrix( (v_dab.T*(da2cc*np.dot(hk, (da2cc.T*(v_dab*dm.reshape(n*n))))) ).reshape((n,n)))
  return vh_coo

