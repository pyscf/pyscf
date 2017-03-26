import numpy
from pyscf.nao.m_siesta_hsx_bloch_mat import siesta_hsx_bloch_mat
#
#
#
def sv_get_denmat(sv, mattype='hamiltonian', prec=64, kvec=[0.0,0.0,0.0], spin=0):

  assert(type(prec)==int)

  mat = None
  mtl = mattype.lower()
  
  if(sv.hsx.is_gamma):

    mat = numpy.empty((sv.norbs,sv.norbs), dtype='float'+str(prec), order='F')
    if(mtl=='hamiltonian'):
      mat = sv.hsx.spin2h4_csr[spin].todense()
    elif(mtl=='overlap'):
      mat = sv.hsx.s4_csr.todense()
    else: 
      raise SystemError('!mattype')

  else:
    if(mtl=='hamiltonian'):
      mat = siesta_hsx_bloch_mat(sv.hsx.spin2h4_csr[spin], sv.hsx, prec=prec, kvec=kvec)
    elif(mtl=='overlap'):
      mat = siesta_hsx_bloch_mat(sv.hsx.s4_csr, sv.hsx, prec=prec, kvec=kvec)
    else:
      raise SystemError('!mattype')

  return mat
