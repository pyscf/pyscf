#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf

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
m.conv_tol = 1e-9
m.scf()


class KnowValues(unittest.TestCase):
    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mc.fcisolver = dmrgscf.CheMPS2(mol)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc2step_6o6e(self):
        mc = mcscf.CASSCF(m, 6, 6)
        mc.fcisolver = dmrgscf.CheMPS2(mol)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()

