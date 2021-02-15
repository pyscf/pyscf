from __future__ import print_function, division
import unittest
import numpy as np
from pyscf.nao import nao
from os.path import dirname, abspath

class KnowValues(unittest.TestCase):

  def test_vna_n2(self):
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

  def test_vna_lih(self):
    dname = dirname(abspath(__file__))
    n = nao(label='lih', cd=dname)
    m = 200
    dvec,midv = 2*(n.atom2coord[1] - n.atom2coord[0])/m,  (n.atom2coord[1] + n.atom2coord[0])/2.0
    vgrid = np.tensordot(np.array(range(-m,m+1)), dvec, axes=0) + midv
    sgrid = np.array(range(-m,m+1)) * np.sqrt((dvec*dvec).sum())
    
    
    #vgrid = np.array([[-1.517908564663352e+00, 1.180550033093826e+00,0.000000000000000e+00]])
    vna = n.vna(vgrid)
    
    #for v,r in zip(vna,vgrid):
    #  print("%23.15e %23.15e %23.15e %23.15e"%(r[0], r[1], r[2], v))
    
    #print(vna.shape, sgrid.shape)
    np.savetxt('vna_lih_0004.txt', np.row_stack((sgrid, vna)).T)
    ref = np.loadtxt(dname+'/vna_lih_0004.txt-ref')
    for r,d in zip(ref[:,1],vna): self.assertAlmostEqual(r,d)

  def test_water_vkb(self):
    """ This  """
    from numpy import einsum, array
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    sv = nao(label='water', cd=dname)


    
if __name__ == "__main__": unittest.main()
