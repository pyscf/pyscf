from __future__ import print_function, division
import numpy
from scipy.sparse import csr_matrix
from numpy import empty 

#
#
#
def siesta_hsx_bloch_mat_vecs(csr, hsx, kvec=[0.0,0.0,0.0], prec=64):
  assert(type(prec)==int)
  
  if(csr.nnz!=hsx.x4.shape[0]): 
    print(csr.nnz, hsx.x4.shape)
    raise SystemError('!nnz')

  if(hsx.norbs!=len(csr.indptr)-1): raise SystemError('!csr')

  den_bloch = numpy.zeros((hsx.norbs,hsx.norbs), dtype='complex'+str(2*prec))

  for row in range(hsx.norbs):
    s,f = csr.indptr[row],csr.indptr[row+1]
    rvecs = hsx.x4[s:f,:]
    cols  = hsx.orb_sc2orb_uc[csr.indices[s:f]]
    phase_facs = numpy.exp(1.0j*numpy.dot(rvecs,kvec))
    den_bloch[row,cols]=den_bloch[row,cols]+csr.data[s:f]*phase_facs

  return den_bloch

#
#
#
def siesta_hsx_bloch_mat(csr, hsx, kvec=[0.0,0.0,0.0], prec=64):
  assert(type(prec)==int)
  
  if(csr.nnz!=hsx.x4.shape[0]): 
    print(csr.nnz, hsx.x4.shape)
    raise SystemError('!nnz')

  if(hsx.norbs!=len(csr.indptr)-1): raise SystemError('!csr')

  den_bloch = numpy.zeros((hsx.norbs,hsx.norbs), dtype='complex'+str(2*prec))

  for row in range(hsx.norbs):
    for ind in range(csr.indptr[row],csr.indptr[row+1]):
      rvec = hsx.x4[ind,:]
      col_sc = csr.indices[ind]
      col = hsx.orb_sc2orb_uc[col_sc]
      phase_fac = numpy.exp(1.0j*numpy.dot(rvec,kvec))
      den_bloch[col,row]=den_bloch[col,row]+csr.data[ind]*phase_fac

  return den_bloch


