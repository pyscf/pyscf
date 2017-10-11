from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_openmx(self):
    """ Computing of the atomic orbitals """
    from pyscf.nao import system_vars_c
   
    sv = system_vars_c().init_openmx(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    self.assertEqual(sv.natoms, 3)
    self.assertEqual(sv.norbs, 23)
    
if __name__ == "__main__": unittest.main()
