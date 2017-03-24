import numpy
from scipy.sparse import csr_matrix
from numpy import empty 

#
#
#
def _siesta2blanko_csr(orb2m, mat, orb_sc2orb_uc=None):
  #print(' m    ',  len(orb2m))
  #print(' data ',  len(mat.data))
  #print(' col  ',  len(mat.indices))
  #print(' ptr  ',  len(mat.indptr))
  n = len(orb2m)
  if(n!=len(mat.indptr)-1): raise SystemError('!mat')
  for row in range(n):
    m1  = orb2m[row]
    for ind in range(mat.indptr[row],mat.indptr[row+1]):
      col = mat.indices[ind]
      if(col>=n): col=orb_sc2orb_uc[col]
      m2 = orb2m[col]
      v  = mat.data[ind]*(-1.0)**m1 * (-1.0)**m2
      mat.data[ind] = v
  return(0)
