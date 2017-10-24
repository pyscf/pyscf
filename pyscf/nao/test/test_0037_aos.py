from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_aos_libnao(self):
    """ Computing of the atomic orbitals """
    from pyscf.nao import system_vars_c
    from pyscf.tools.cubegen import Cube
   
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    cc = Cube(sv, nx=20, ny=20, nz=20)
    aos = sv.comp_aos_den(cc.get_coords())
    self.assertEqual(aos.shape[0], cc.nx*cc.ny*cc.nz)
    self.assertEqual(aos.shape[1], sv.norbs)

if __name__ == "__main__": unittest.main()
