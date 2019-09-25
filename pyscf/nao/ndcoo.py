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
    '''converts shape of sparse matrix () into (p*a, b)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[0]+self.ind[1]*s[0], self.ind[2])) )

  def tocoo_p_ab(self, descr):
    '''converts shape of sparse matrix () into (p, a*b)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[0], self.ind[2]+self.ind[1]*s[2])) )

  def tocoo_a_pb(self, descr):
    '''converts shape of sparse matrix () into (a, p*b)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[1], self.ind[0]+self.ind[2]*s[0])) )      

  def tocoo_b_pa(self, descr):
    '''converts shape of sparse matrix () into (b, p*a)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[2], self.ind[0]+self.ind[1]*s[0])) )  

if __name__=='__main__':
  import numpy as np
  ref = np.random.rand(70,12,35)
  #ref = np.array([[[1, 2], [0, 0], [0, 0], [3, 4]],[[0, 0], [1, 0], [5 ,6], [0, 1]], [[0, 9], [7, 8], [0, 0], [10, 0]]])
  print('ref shape: ====>\t',ref.shape)
  data = ref.reshape(-1)  #collect all data as 1D
  i0,i1,i2 = np.mgrid[0:ref.shape[0],0:ref.shape[1],0:ref.shape[2] ].reshape((3,data.size)) #provides mesh for i1,... which are shapes of given matrix
  
  #print(data.shape, i0.shape)
  #print(data)
  #print(i0,i1,i2)
  nc = ndcoo((data, (i0, i1, i2)))        #gives inputs to class ndcoo



  m0 = nc.tocoo_pa_b('p,a,b->ap,b')                 #change to favorable shape (ap,b) in COO format
  print('reshaped and sparse matrix m0(pa,b): ====>\t',m0.shape)             #(ap,b)            
  m0 = nc.tocoo_pa_b('p,a,b->ap,b').toarray().reshape((nc.shape[1], nc.shape[0], nc.shape[2]))
  m0 = np.swapaxes(m0,0,1)
  print('m0 reshaped to 3D array (a ,p ,b)',m0.shape) 
  print('comparison between ref and m0: ====>\t ',np.allclose(m0, ref)) #decompressed, reshaped and swapped matrix m0 should be equal to ref
                                          

  m1 = nc.tocoo_p_ab('p,a,b->p,ba')
  print('reshaped and sparse matrix m1(p,ba): ====>\t',m1.shape) 
  m1 = nc.tocoo_p_ab('p,a,b->p,ba').toarray().reshape((nc.shape[0], nc.shape[1], nc.shape[2]))
  print('m1 reshaped to 3D array ( p, a, b)',m1.shape)
  print('comparison between ref and m1: ====>\t ', np.allclose(m1, ref))     #compressed and reshaped matrix should be equal to referance array



  m2 = nc.tocoo_a_pb('p,a,b->a,pb')
  print('reshaped and sparse matrix m2(a,pb): ====>\t',m2.shape) 
  m2 = nc.tocoo_a_pb('p,a,b->a,pb').toarray().reshape((nc.shape[1], nc.shape[2], nc.shape[0]))
  m2 = np.swapaxes(m2.T,1,2)
  print('m2 reshaped to 3D array (p ,a , b)',m2.shape)
  print('comparison between ref and m2: ====>\t ', np.allclose(m2, ref))


  m3 = nc.tocoo_b_pa('p,a,b->b,pa')
  print('reshaped and sparse matrix m3(b,pa): ====>\t',m3.shape) 
  m3 = nc.tocoo_b_pa('p,a,b->b,pa').toarray().reshape((nc.shape[2], nc.shape[1], nc.shape[0]))
  m3 = m3.T
  print('m3 reshaped to 3D array (p ,a , b)',m3.shape)
  print('comparison between ref and m3: ====>\t ', np.allclose(m3, ref))


    # Constructing a matrix using ijv format
    #row  = np.array([0, 3, 1, 0])
    #col  = np.array([0, 3, 1, 2])
    #data = np.array([4, 5, 7, 9])
    #coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    #array([[4, 0, 9, 0],
    #       [0, 7, 0, 0],
    #       [0, 0, 0, 0],
    #       [0, 0, 0, 5]])
