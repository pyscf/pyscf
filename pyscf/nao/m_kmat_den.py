from __future__ import print_function, division

def kmat_den(sv, dm=None, algo='fci', **kvargs):
  """
  Computes the matrix elements of Fock exchange operator
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from pyscf.nao.m_prod_basis import prod_basis_c
  from scipy.sparse import csr_matrix
  import numpy as np

  pb,hk=sv.add_pb_hk(**kvargs)
  dm = sv.comp_dm(**kvargs) if dm is None else dm  
  algol = algo.lower()
  if algol=='fci':
    sv.fci_den = abcd2v = sv.fci_den if hasattr(sv, 'fci_den') else pb.comp_fci_den(hk)
    kmat = np.einsum('abcd,bc->ad', abcd2v, dm)
  else:
    print('algo=', algo)
    raise RuntimeError('unknown algoright')
  
  return kmat

