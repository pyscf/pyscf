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
h2o.grids = {"H": (10, 50),
             "O": (10, 50),}
h2o.build()

class KnowValues(unittest.TestCase):
    def test_gen_grid(self):
        grid = gen_grid.Grids(h2o)
        grid.becke_scheme = gen_grid.original_becke
        grid.atomic_radii = numpy.round(dft.radi.BRAGG_RADII, 2)
        coord, weight = grid.setup_grids()
        self.assertAlmostEqual(numpy.linalg.norm(coord), 185.91245945279027, 9)
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1720.1317185648893, 9)

        grid.becke_scheme = gen_grid.stratmann
        coord, weight = grid.setup_grids()
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1730.3692983091271, 9)

    def test_radi(self):
        grid = gen_grid.Grids(h2o)
        grid.atomic_radii = radi.COVALENT_RADII
        grid.radi_method = radi.mura_knowles
        coord, weight = grid.setup_grids()
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1795.8808603796606, 9)

        grid.radi_method = radi.delley
        coord, weight = grid.setup_grids()
        self.assertAlmostEqual(numpy.linalg.norm(weight), 1676.730797573287, 9)


if __name__ == "__main__":
    print("Test Grids")
    unittest.main()


