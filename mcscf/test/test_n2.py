#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf

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
m.conv_threshold = 1e-9
m.scf()

molsym = gto.Mole()
molsym.build(
verbose = 0,
output = None,
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
symmetry = True
)
msym = scf.RHF(molsym)
msym.conv_threshold = 1e-9
msym.scf()


class KnowValues(unittest.TestCase):
    def test_mc1step_4o4e(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        emc = mc.mc1step()[0] + mol.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        emc = mc.mc2step()[0] + mol.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc1step_6o6e(self):
        mc = mcscf.CASSCF(mol, m, 6, 6)
        emc = mc.mc1step()[0] + mol.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_6o6e(self):
        mc = mcscf.CASSCF(mol, m, 6, 6)
        emc = mc.mc2step()[0] + mol.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc1step_symm_4o4e(self):
        mc = mcscf.CASSCF(molsym, msym, 4, 4)
        emc = mc.mc1step()[0] + molsym.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc2step_symm_4o4e(self):
        mc = mcscf.CASSCF(molsym, msym, 4, 4)
        emc = mc.mc2step()[0] + molsym.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc1step_symm_6o6e(self):
        mc = mcscf.CASSCF(molsym, msym, 6, 6)
        emc = mc.mc1step()[0] + molsym.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_symm_6o6e(self):
        mc = mcscf.CASSCF(molsym, msym, 6, 6)
        emc = mc.mc2step()[0] + molsym.nuclear_repulsion()
        self.assertAlmostEqual(emc, -108.980105451388, 7)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()

