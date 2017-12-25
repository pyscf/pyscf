from __future__ import print_function, division

def kmat_den(mf, dm=None, algo='fci', **kw):
  """
  Computes the matrix elements of Fock exchange operator
  Args:
    mf : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from pyscf.nao.m_prod_basis import prod_basis_c
  from scipy.sparse import csr_matrix
  import numpy as np

  pb,hk=mf.add_pb_hk(**kw)
  dm = mf.make_rdm1() if dm is None else dm

  n = mf.norbs
  if mf.nspin==1:
    dm = dm.reshape((n,n))
  elif mf.nspin==2:
    dm = dm.reshape((mf.nspin,n,n))
  else:
    print(nspin)
    raise RuntimeError('nspin>2?')
    
  algol = algo.lower()
  if algol=='fci':
    mf.fci_den = abcd2v = mf.fci_den if hasattr(mf, 'fci_den') else pb.comp_fci_den(hk)
    kmat = np.einsum('abcd,...bc->...ad', abcd2v, dm)
  else:
    print('algo=', algo)
    raise RuntimeError('unknown algorithm')

  return kmat

