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
import unittest
from pyscf.nao import system_vars_c 
from pyscf.nao.m_gauleg import gauss_legendre
from pyscf import dft
import numpy as np

class KnowValues(unittest.TestCase):

  def test_bilocal(self):
    """ Build 3d integration scheme for two centers."""
    sv=system_vars_c().init_xyzlike([ [8, [0.0, 0.0, 0.0]], [1, [1.0, 1.0, 1.0] ]])
    atom2rcut=np.array([5.0, 4.0])
    grids = dft.gen_grid.Grids(sv)
    grids.level = 2 # precision as implemented in pyscf
    grids.radi_method = gauss_legendre
    grids.build(atom2rcut=atom2rcut)
    self.assertEqual(len(grids.weights), 20648)

  def test_one_center(self):
    """ Build 3d integration coordinates and weights for just one center. """
    sv=system_vars_c().init_xyzlike([ [8, [0.0, 0.0, 0.0]]])
    atom2rcut=np.array([5.0])
    g = dft.gen_grid.Grids(sv)
    g.level = 1 # precision as implemented in pyscf
    g.radi_method = gauss_legendre
    g.build(atom2rcut=atom2rcut)

    #print(  max(  np.linalg.norm(g.coords, axis=1)  )  )
    #print(  g.weights.sum(), 4.0 *np.pi*5.0**3 / 3.0 )
    self.assertAlmostEqual(max(  np.linalg.norm(g.coords, axis=1)  ), 4.9955942742763986)
    self.assertAlmostEqual(g.weights.sum(), 4.0 *np.pi*5.0**3 / 3.0)
    self.assertEqual(len(g.weights), 6248)
    

if __name__ == "__main__":
  unittest.main()
