import numpy
from scipy.sparse import csr_matrix
from numpy import empty 

#
#
#
def siesta_hsx_bloch_mat(csr, hsx, kvec=[0.0,0.0,0.0], prec=64):
  assert(type(prec)==int)
  
  if(csr.nnz!=hsx.x4.shape[1]): raise SystemError('!nnz')
  if(hsx.norbs!=len(csr.indptr)-1): raise SystemError('!csr')

  den_bloch = numpy.zeros((hsx.norbs,hsx.norbs), dtype='complex'+str(2*prec), order='F')
  
  for row in range(hsx.norbs):
    for ind in range(csr.indptr[row],csr.indptr[row+1]):
      rvec = hsx.x4[0:3,ind]
      col_sc = csr.indices[ind]
      col = hsx.orb_sc2orb_uc[col_sc]
      phase_fac = numpy.exp(1.0j*numpy.dot(rvec,kvec))
      den_bloch[col,row]=den_bloch[col,row]+csr.data[ind]*phase_fac

  return(den_bloch)
