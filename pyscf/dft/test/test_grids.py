#!/usr/bin/env python
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

import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf.dft import gen_grid
from pyscf.dft import radi

h2o = gto.Mole()
h2o.verbose = 5
h2o.output = '/dev/null'
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.basis = {"H": '6-31g',
             "O": '6-31g',}
h2o.build()

def tearDownModule():
    global h2o
    h2o.stdout.close()
    del h2o


class KnownValues(unittest.TestCase):
    def test_gen_grid(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = None
        grid.radi_method = radi.gauss_chebyshev
        grid.becke_scheme = gen_grid.original_becke
        grid.radii_adjust = radi.becke_atomic_radii_adjust
        grid.atomic_radii = radi.BRAGG_RADII
        grid.atom_grid = {"H": (10, 50), "O": (10, 50),}
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.coords), 185.91245945279027, 9)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 1720.1317185648893, 8)

        grid.becke_scheme = gen_grid.stratmann
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 1730.3692983091271, 8)

        grid.atom_grid = {"O": (10, 50),}
        grid.radii_adjust = None
        grid.becke_scheme = gen_grid.stratmann
        grid.kernel(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 2559.0064040257907, 8)

        grid.atom_grid = (10, 11)
        grid.becke_scheme = gen_grid.original_becke
        grid.radii_adjust = None
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 1712.3069450297105, 8)

    def test_radi(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = None
        grid.radii_adjust = radi.becke_atomic_radii_adjust
        grid.atomic_radii = radi.COVALENT_RADII
        grid.radi_method = radi.mura_knowles
        grid.atom_grid = {"H": (10, 50), "O": (10, 50),}
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 1804.5437331817291, 9)

        grid.radi_method = radi.delley
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 1686.3482864673697, 9)

        grid.radi_method = radi.becke
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 45009387.132578261, 7)

    def test_prune(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = gen_grid.sg1_prune
        grid.atom_grid = {"H": (10, 50), "O": (10, 50),}
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.coords), 202.17732600266302, 9)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 442.54536463517167, 9)

        grid.prune = gen_grid.nwchem_prune
        grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(grid.coords), 149.55023044392638, 9)
        self.assertAlmostEqual(numpy.linalg.norm(grid.weights), 586.36841824004455, 9)

        z = 16
        rad, dr = radi.gauss_chebyshev(50)
        angs = gen_grid.sg1_prune(z, rad, 434, radii=radi.SG1RADII)
        self.assertAlmostEqual(lib.finger(angs), -291.0794420982329, 9)

        angs = gen_grid.nwchem_prune(z, rad, 434, radii=radi.BRAGG_RADII)
        self.assertAlmostEqual(lib.finger(angs), -180.12023039394498, 9)

        angs = gen_grid.nwchem_prune(z, rad, 26, radii=radi.BRAGG_RADII)
        self.assertTrue(numpy.all(angs==26))

    def test_gen_atomic_grids(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = None
        grid.atom_grid = {"H": (10, 58), "O": (10, 50),}
        self.assertRaises(ValueError, grid.build)

    def test_make_mask(self):
        grid = gen_grid.Grids(h2o)
        grid.atom_grid = {"H": (10, 110), "O": (10, 110),}
        grid.build()
        coords = grid.coords*10.
        non0 = gen_grid.make_mask(h2o, coords)
        self.assertEqual(non0.sum(), 106)
        self.assertAlmostEqual(lib.finger(non0), -0.81399929716237085, 9)



if __name__ == "__main__":
    print("Test Grids")
    unittest.main()

