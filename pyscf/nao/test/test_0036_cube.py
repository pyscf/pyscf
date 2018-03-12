# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf import gto, scf
mol = gto.M(atom='O 0.00000000,  0.000000,  0.000000; H 0.761561, 0.478993, 0.00000000,; H -0.761561, 0.478993, 0.00000000,', basis='6-31g*', verbose=0)
mf = scf.RHF(mol)
mf.scf()


class KnowValues(unittest.TestCase):

  def test_cubegen(self):
    """ Compute the density and store into a cube file  """
    from pyscf.tools import cubegen
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1(), nx=20, ny=20, nz=20)
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1(), nx=20, ny=20, nz=20) #slow

  def test_cube_c(self):
    """ Compute the density and store into a cube file  """
    from pyscf.tools.cubegen import Cube
    from pyscf.dft import numint, gen_grid

    # Initialize the class cube_c
    cc = Cube(mol, nx=20, ny=20, nz=20)
    
    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = np.empty(ngrids)
    ao = None
    dm = mf.make_rdm1()
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(cc.nx,cc.ny,cc.nz)
    
    # Write out density to the .cube file
    cc.write(rho, "h2o_den_cube_c.cube", comment='Electron density in real space (e/Bohr^3)')

  def test_cube_sv(self):
    """ Compute the density and store into a cube file  """
    from pyscf.nao import system_vars_c
    from pyscf.nao.m_comp_dm import comp_dm
    from pyscf.tools.cubegen import Cube
    
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    cc = Cube(sv, nx=50, ny=50, nz=50)
    dens = sv.dens_elec(cc.get_coords(), sv.comp_dm())
    dens = dens[:,0].reshape(cc.nx,cc.ny,cc.nz)
    cc.write(dens, "water.cube", comment='Valence electron density in real space (e/Bohr^3)')

    self.assertEqual(len(cc.xs), cc.nx)
    self.assertEqual(len(cc.ys), cc.ny)
    self.assertEqual(len(cc.zs), cc.nz)
    self.assertAlmostEqual(dens.sum()*cc.get_volume_element(), 8.0, 1)
    

if __name__ == "__main__": unittest.main()
