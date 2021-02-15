from __future__ import print_function, division
import unittest, numpy as np
import os
from pyscf.nao import mf as mf_c 

class KnowValues(unittest.TestCase):

  def test_double_sparse(self):
    """ This is a test of a double-sparse storage of the vertex """
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='water', cd=dname)
    pb = mf.pb
    v_dab_array = pb.get_dp_vertex_array()
    nnn = v_dab_array.size
    vds = pb.get_dp_vertex_doubly_sparse(axis=0)
    self.assertEqual(vds.shape, v_dab_array.shape)
    self.assertTrue(abs(vds.toarray()-v_dab_array).sum()/nnn<1e-14)
    vds = pb.get_dp_vertex_doubly_sparse(axis=1)
    self.assertTrue(abs(vds.toarray()-v_dab_array).sum()/nnn<1e-14)
    vds = pb.get_dp_vertex_doubly_sparse(axis=2)
    self.assertTrue(abs(vds.toarray()-v_dab_array).sum()/nnn<1e-14)
    
        

if __name__ == "__main__": unittest.main()
