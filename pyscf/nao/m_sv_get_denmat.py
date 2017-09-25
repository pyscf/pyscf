from __future__ import print_function, division
import numpy
from pyscf.nao.m_siesta_hsx_bloch_mat import siesta_hsx_bloch_mat
from timeit import default_timer as timer
#
#
#
def sv_get_denmat(sv, mattype='hamiltonian', prec=64, kvec=[0.0,0.0,0.0], spin=0):

  assert(type(prec)==int)

  mat = None
  mtl = mattype.lower()
  
  if(sv.hsx.is_gamma):

    mat = numpy.empty((sv.norbs,sv.norbs), dtype='float'+str(prec))
    if(mtl=='hamiltonian'):
      mat = sv.hsx.spin2h4_csr[spin].todense()
    elif(mtl=='overlap'):
      mat = sv.hsx.s4_csr.todense()
    else: 
      raise SystemError('!mattype')

  else:
    #t1 = timer()
    if(mtl=='hamiltonian'):
      mat = siesta_hsx_bloch_mat(sv.hsx.spin2h4_csr[spin], sv.hsx, kvec=kvec)
    elif(mtl=='overlap'):
      mat = siesta_hsx_bloch_mat(sv.hsx.s4_csr, sv.hsx, kvec=kvec)
    else:
      raise SystemError('!mattype')
    #t2 = timer(); print(t2-t1)
    
  return mat
