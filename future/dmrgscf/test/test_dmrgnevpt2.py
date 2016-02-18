#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf
from pyscf.mrpt import nevpt2
dmrgscf.settings.MPIPREFIX = 'mpirun -n 4'

b = 1.4
mol = gto.M(verbose = 0,
atom = [
    ['N', (0, 0, -b/2)],
    ['N', (0, 0,  b/2)], ],
basis = '631g')
m = scf.RHF(mol)
m.conv_tol = 1e-12
m.scf()

mc = dmrgscf.dmrgci.DMRGSCF(m, 4, 4)
mc.kernel()


class KnowValues(unittest.TestCase):
    def test_nevpt2_with_4pdm(self):
        e = nevpt2.sc_nevpt(mc)
        self.assertAlmostEqual(e, -0.14058373193902649, 6)

    def test_nevpt2_without_4pdm(self):
        e = nevpt2.sc_nevpt(dmrgscf.compress_perturb(mc, maxM=100))
        self.assertAlmostEqual(e, -0.14058405242161856, 6)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()


