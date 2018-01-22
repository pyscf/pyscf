from __future__ import print_function, division
import unittest,os


class KnowValues(unittest.TestCase):

  def test_vrtx_cc_apairs(self):
    """ This is to test a batch generation vertices for bilocal atomic pairs. """
    from pyscf.nao import mf, prod_basis
    from numpy import allclose

    dname = os.path.dirname(os.path.abspath(__file__))
    sv = mf(label='water', cd=dname)
    pbb = sv.pb
    pba = prod_basis(sv)

    for a,b in zip(pba.bp2info,pbb.bp2info):
      for a1,a2 in zip(a.atoms,b.atoms): self.assertEqual(a1,a2)
      for a1,a2 in zip(a.cc2a, b.cc2a): self.assertEqual(a1,a2)
      self.assertTrue(allclose(a.vrtx, b.vrtx))
      self.assertTrue(allclose(a.cc, b.cc))

    self.assertLess(abs(pbb.get_da2cc_sparse().tocsr()-pba.get_da2cc_sparse().tocsr()).sum(), 1e-9)
    self.assertLess(abs(pbb.get_dp_vertex_sparse().tocsr()-pba.get_dp_vertex_sparse().tocsr()).sum(), 1e-10)
    
if __name__ == "__main__": unittest.main()
