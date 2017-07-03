from __future__ import print_function, division
import unittest

class KnowValues(unittest.TestCase):

  def test_water_pp_pb(self):
    """ This is for initializing with SIESTA radial orbitals """
    from pyscf.nao import system_vars_c, prod_basis_c
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(chdir=dname)
    self.assertTrue(abs(sv.ucell).sum()>0)
    pb = prod_basis_c().init_pb_pp_libnao_apair(sv)
    self.assertEqual(sv.norbs, 23)
    ap = pb.comp_apair_pp_libint(0,1)
    pb.init_pb_pp_all_pairs()
    self.assertEqual(len(pb.bp2info), 3)

if __name__ == "__main__":
  unittest.main()
