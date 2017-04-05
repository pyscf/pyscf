#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf.dft import gen_grid
from pyscf.dft import radi

h2o = gto.Mole()
h2o.verbose = 0
h2o.output = None#"out_h2o"
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.basis = {"H": '6-31g',
             "O": '6-31g',}
h2o.build()

class KnowValues(unittest.TestCase):
    def test_gen_grid(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = None
        grid.radi_method = radi.gauss_chebyshev
        grid.becke_scheme = gen_grid.original_becke
        grid.radii_adjust = radi.becke_atomic_radii_adjust
        grid.atomic_radii = radi.BRAGG_RADII
        grid.atom_grid = {"H": (10, 50), "O": (10, 50),}
        coord, weight = grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(coord), 185.91245945279027, 9)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1720.1317185648893, 8)

        grid.becke_scheme = gen_grid.stratmann
        coord, weight = grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1730.3692983091271, 8)

        grid.atom_grid = {"O": (10, 50),}
        grid.radii_adjust = None
        grid.becke_scheme = gen_grid.stratmann
        coord, weight = grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 2559.006415248321, 8)

    def test_radi(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = None
        grid.radii_adjust = radi.becke_atomic_radii_adjust
        grid.atomic_radii = radi.COVALENT_RADII
        grid.radi_method = radi.mura_knowles
        grid.atom_grid = {"H": (10, 50), "O": (10, 50),}
        coord, weight = grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1804.5437331817291, 9)

        grid.radi_method = radi.delley
        coord, weight = grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1686.3482864673697, 9)

    def test_prune(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = gen_grid.sg1_prune
        grid.atom_grid = {"H": (10, 50), "O": (10, 50),}
        coord, weight = grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(coord), 202.17732600266302, 9)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 442.54536463517167, 9)

        grid.prune = gen_grid.nwchem_prune
        coord, weight = grid.build(with_non0tab=False)
        self.assertAlmostEqual(numpy.linalg.norm(coord), 149.55023044392638, 9)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 586.36841824004455, 9)

    def test_gen_atomic_grids(self):
        grid = gen_grid.Grids(h2o)
        grid.prune = None
        grid.atom_grid = {"H": (10, 58), "O": (10, 50),}
        self.assertRaises(ValueError, grid.build)


if __name__ == "__main__":
    print("Test Grids")
    unittest.main()

