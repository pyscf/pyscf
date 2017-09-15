from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_mo_cube_libnao(self):
    """ Computing of the atomic orbitals """
    from pyscf.nao import system_vars_c
    from pyscf.tools.m_cube import cube_c
   
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    cc = cube_c(sv, nx=20, ny=20, nz=20)
    co2val = sv.comp_aos_den(cc.get_coords())
    c2orb =  np.dot(co2val, sv.wfsx.x[0,0,0,:,0]).reshape((cc.nx, cc.ny, cc.nz))
    cc.write(c2orb, "water_mo.cube", comment='Molecular orbital')
    
if __name__ == "__main__": unittest.main()
