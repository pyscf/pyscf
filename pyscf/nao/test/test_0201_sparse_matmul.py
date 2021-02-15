from __future__ import print_function, division
import unittest, numpy as np
import scipy.sparse as sprs
from timeit import default_timer as timer

class KnowValues(unittest.TestCase):

  def test_0201_sparse_matmul(self):
    """ The testbed for checking different ways of invoking matrix-matrix multiplications """

    return
    
    for n in [50, 100, 200, 400, 800, 1600, 3200]:
      print()
      for dens in [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]:
      
        asp = sprs.random(n, n, format='csr', density=dens)
        bsp = sprs.random(n, n, format='csr', density=dens)

        t1 = timer()
        cmat1 = np.dot(asp, bsp)
        t2 = timer(); ts =t2-t1; #print('runtime sparse ', ts)

    
        adn = asp.toarray()
        bdn = bsp.toarray()
        t1 = t2
        cmat2 = np.dot(adn, bdn)
        t2 = timer(); td =t2-t1; #print('runtime  dense ', td) 
        t1 = t2
    
        print('dens, ratio {:5d}, {:.6f} {:.6f} {:.6f} {:.6f}'.format(n, dens, td, ts, td/ts))
    
    
if __name__ == "__main__": unittest.main()
