#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf
dmrgscf.settings.MPIPREFIX = 'mpirun -n 4'

b = 1.4
mol = gto.Mole()
mol.build(
verbose = 0,
output = None,
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
)
m = scf.RHF(mol)
m.conv_tol = 1e-12
m.scf()


class KnowValues(unittest.TestCase):
    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mc.fcisolver = dmrgscf.DMRGCI(mol)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc1step_4o4e(self):
        mc = dmrgscf.DMRGSCF(m, 4, 4)
        emc = mc.kernel()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()

