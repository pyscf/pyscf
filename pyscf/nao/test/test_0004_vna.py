from __future__ import print_function, division
import unittest
import numpy as np
from pyscf.nao import nao
from os.path import dirname, abspath

class KnowValues(unittest.TestCase):

  def test_vna(self):
    dname = dirname(abspath(__file__))
    n = nao(label='n2', cd=dname)
    m = 200
    dvec,midv = 2*(n.atom2coord[1] - n.atom2coord[0])/m,  (n.atom2coord[1] + n.atom2coord[0])/2.0
    vgrid = np.tensordot(np.array(range(-m,m+1)), dvec, axes=0) + midv
    sgrid = np.array(range(-m,m+1)) * np.sqrt((dvec*dvec).sum())
    
    vna = n.vna(vgrid)
    #print(vna.shape, sgrid.shape)
    #np.savetxt('vna_n2_0004.txt', np.row_stack((sgrid, vna)).T)
    ref = np.loadtxt(dname+'/vna_n2_0004.txt-ref')
    for r,d in zip(ref[:,1],vna): self.assertAlmostEqual(r,d)
    
if __name__ == "__main__": unittest.main()
