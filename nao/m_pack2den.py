import numpy as np

#
#
#
def pack2den(pack):
  """
    Unpacks packed format to dense format 
  """
  dim = size2dim(len(pack))
  den = np.zeros((dim,dim))
  for i in range(dim):
    for j in range(i+1):
      den[i,j] = pack[i*(i+1)//2+j]
      den[j,i] = den[i,j]
  return den
    
#
#
#
def size2dim(pack_size):
  """
    Computes dimension of the matrix by it's packed size. Packed size is n*(n+1)//2
  """
  rdim = (np.sqrt(8.0*pack_size+1.0) - 1.0)/2.0
  ndim = int(rdim)
  if ndim != rdim : raise SystemError('!ndim/=rdim')
  if ndim*(ndim+1)//2 != pack_size: SystemError('!pack_size')
  return ndim

#
#
#
def ij2pack(i,j):
  ma = max(i,j)
  return ma*(ma+1)//2+min(i,j)
