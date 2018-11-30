from __future__ import print_function, division
import numpy as np
from scipy.sparse import coo_matrix
from timeit import default_timer as timer

#
#
#
class ndcoo():
  '''
  '''
  def __init__(self, inp, **kw):
    
    self.data = inp[0]
    self.ind = inp[1]
    self.shape = (self.ind[0].max()+1, self.ind[1].max()+1, self.ind[2].max()+1)
    self.ndim = len(self.ind)
    self.dtype = np.float64
  
  def tocoo(self, descr):
    
    s = self.shape
    return coo_matrix((self.data, (self.ind[0]+self.ind[1]*s[0], self.ind[2])) )

    

if __name__=='__main__':
  import numpy as np

  ref = np.random.rand(3*4*2).reshape((3, 4, 2))
  data = ref.reshape(-1)
  i0,i1,i2 = np.mgrid[0:ref.shape[0],0:ref.shape[1],0:ref.shape[2] ].reshape((3,data.size))
  
  print(data.shape, i0.shape)
  print(i0)
  nc = ndcoo((data, (i0, i1, i2)))
  
  m = nc.tocoo('p,a,b->ap,b').toarray().reshape((nc.shape[1], nc.shape[0], nc.shape[2]))
  #mcsr.nc.tocoo('p,a,b->ap,b').tocsr()
  #m = nc.tocoo('p,a,b->ap,b').toarray()
  print(m.shape)
  print(ref.shape)
  print(np.allclose(m, np.swapaxes(ref, 0, 1)))
  
  
  
  
