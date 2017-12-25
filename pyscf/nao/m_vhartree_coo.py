from __future__ import print_function, division
import numpy as np

def dm2j_fullmat(n, v_dab, da2cc, hk, dm):
  return (v_dab.T*(da2cc*np.dot(hk, (da2cc.T*(v_dab*dm.reshape(n*n))))) ).reshape((n,n))

def vhartree_coo(mf, dm=None, **kw):
  """
  Computes the matrix elements of Hartree potential
  Args:
    mf: this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from scipy.sparse import coo_matrix, csr_matrix

  pb,hk = mf.add_pb_hk(**kw)
  dm = mf.make_rdm1() if dm is None else dm
  v_dab = pb.get_dp_vertex_sparse(sparseformat=csr_matrix)
  da2cc = pb.get_da2cc_sparse(sparseformat=csr_matrix)

  n = mf.norbs
  nspin = mf.nspin
  if mf.nspin==1:
    dm = dm.reshape((n,n))
    vh_coo = coo_matrix( dm2j_fullmat(n, v_dab, da2cc, hk, dm) )
  elif mf.nspin==2:
    dm = dm.reshape((nspin,n,n))
    vh_coo = [coo_matrix( dm2j_fullmat(n, v_dab, da2cc, hk, dm[0,:,:]) ),
      coo_matrix( dm2j_fullmat(n, v_dab, da2cc, hk, dm[1,:,:]) )]
  else:
    print(nspin)
    raise RuntimeError('nspin>2?')

  return vh_coo
