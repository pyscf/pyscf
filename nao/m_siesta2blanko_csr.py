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
    col_data = mat.data[mat.indptr[row]:mat.indptr[row+1]]
    col_data = col_data * (-1)**m1
    
    col_phase = numpy.empty((len(col_data)), dtype='int8')
    for ind in range(mat.indptr[row],mat.indptr[row+1]):
      icol = mat.indices[ind]
      if(icol>=n): icol=orb_sc2orb_uc[icol]
      m2 = orb2m[icol]
      col_phase[ind-mat.indptr[row]] = (-1)**m2
    
    col_data = col_data*col_phase
    mat.data[mat.indptr[row]:mat.indptr[row+1]] = col_data
  return(0)
