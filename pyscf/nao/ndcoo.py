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
    '''
    inp: (data, (i1,i2,i3)) when i is the indeces based on the shape of 3D matrix
    '''
    self.data = inp[0]
    self.ind = inp[1]
    self.shape = (self.ind[0].max()+1, self.ind[1].max()+1, self.ind[2].max()+1)
    self.ndim = len(self.ind)
    self.dtype = np.float64
  
  def tocoo_pa_b(self, descr):
    '''converts shape of sparse matrix () into (a*b, c)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[0]+self.ind[1]*s[0], self.ind[2])) )

  def tocoo_p_ab(self, descr):
    '''converts shape of sparse matrix () into (a, b*c)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[0], self.ind[2]+self.ind[1]*s[2])) )      

if __name__=='__main__':
  import numpy as np

  ref = np.array([[[1, 2], [0, 0], [0, 0], [3, 4]],[[0, 0], [1, 0], [5 ,6], [0, 1]], [[0, 9], [7, 8], [0, 0], [10, 0]]])
  print('ref shape: ====>\t',ref,ref.shape)
  data = ref.reshape(-1)  #collect all data as 1D
  i0,i1,i2 = np.mgrid[0:ref.shape[0],0:ref.shape[1],0:ref.shape[2] ].reshape((3,data.size)) #provides mesh for i1,... which are shapes of given matrix
  
  #print(data.shape, i0.shape)
  #print(data)
  #print(i0,i1,i2)
  nc = ndcoo((data, (i0, i1, i2)))  #gives inputs to class ndcoo

  m0 = nc.tocoo_pa_b('p,a,b->ap,b')      #change to favorable shape (ap,b) in COO format
  print('reshaped and sparse matrix m0(pa,b): ====>\t',m0,m0.shape)                         
  m0 = nc.tocoo_pa_b('p,a,b->ap,b').toarray().reshape((nc.shape[1], nc.shape[0], nc.shape[2])) 
  print('comparison between ref and m0: ====>\t ',np.allclose(m0, np.swapaxes(ref, 0, 1))) #compressed and reshaped matrix should be equal to swapped referance
                                                #change to favorable shape (ap,b) in coo, convert to array,reshape it to 3D along a swapping

  m1 = nc.tocoo_p_ab('p,a,b->p,ba')
  print('reshaped and sparse matrix m1(p,ba): ====>\t',m1,m1.shape) 
  m1 = nc.tocoo_p_ab('p,a,b->p,ba').toarray().reshape((nc.shape[0], nc.shape[1], nc.shape[2]))
  print('comparison between ref and m1: ====>\t ', np.allclose(m1, ref))     #compressed and reshaped matrix should be equal to referance array
