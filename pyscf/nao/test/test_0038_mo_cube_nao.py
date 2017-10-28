from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_mo_cube_libnao(self):
    """ Computing of the atomic orbitals """
    from pyscf.nao import scf
    from pyscf.tools.cubegen import Cube
   
    sv = scf(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    cc = Cube(sv, nx=40, ny=40, nz=40)
    co2val = sv.comp_aos_den(cc.get_coords())
    nocc_0t = int(sv.nelectron / 2)
    c2orb =  np.dot(co2val, sv.wfsx.x[0,0,nocc_0t,:,0]).reshape((cc.nx, cc.ny, cc.nz))
    cc.write(c2orb, "water_mo.cube", comment='HOMO')
    
if __name__ == "__main__": unittest.main()
