from __future__ import print_function, division
import numpy
from scipy.sparse import csr_matrix
from numpy import empty 

#
#
#
def siesta_hsx_bloch_mat(csr, hsx, kvec=[0.0,0.0,0.0], prec=64):
  assert(type(prec)==int)
  
  if(csr.nnz!=hsx.x4.shape[0]): 
    print(csr.nnz, hsx.x4.shape)
    raise SystemError('!nnz')

  if(hsx.norbs!=len(csr.indptr)-1): raise SystemError('!csr')

  caux = numpy.exp(1.0j*numpy.dot(hsx.x4,kvec)) * csr.data

  den_bloch = numpy.zeros((hsx.norbs,hsx.norbs), dtype='complex'+str(2*prec))    
  for row in range(hsx.norbs):
    for ind in range(csr.indptr[row], csr.indptr[row+1]):
      col = hsx.orb_sc2orb_uc[csr.indices[ind]]
      den_bloch[col,row]=den_bloch[col,row]+caux[ind]

  return den_bloch


