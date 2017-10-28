from __future__ import print_function, division
from os.path import abspath, dirname
from pyscf.nao import scf
import unittest, numpy as np

sv = scf(label='water', cd=dirname(abspath(__file__)))
pb = sv.pb
vab_arr_ref = pb.get_dp_vertex_array()

class KnowValues(unittest.TestCase):

  def test_pb_doubly_sparse(self):
    """ This is to test generation of a doubly sparse format. """
    self.assertEqual(len(vab_arr_ref.shape), 3)
    self.assertEqual(vab_arr_ref.shape, (pb.npdp,pb.norbs,pb.norbs))
    vab_ds = pb.get_dp_vertex_doubly_sparse(axis=2)
    vab_arr = vab_ds.toarray()
    self.assertLess(abs(vab_arr-vab_arr_ref).sum()/vab_arr.size, 1e-15)

  def test_pb_sparse(self):
    """ This is to test generation of a doubly sparse format. """
    self.assertEqual(len(vab_arr_ref.shape), 3)
    self.assertEqual(vab_arr_ref.shape, (pb.npdp,pb.norbs,pb.norbs))
    vab_s = pb.get_dp_vertex_sparse()
    vab_arr = vab_s.toarray().reshape((pb.npdp,pb.norbs,pb.norbs))
    self.assertLess(abs(vab_arr-vab_arr_ref).sum()/vab_arr.size, 1e-15)
    
if __name__ == "__main__":
  print("Tests sparse tables generation facilities of the class prod_basis_c")
  unittest.main()

