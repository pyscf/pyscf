#!/usr/bin/env python
import unittest
from pyscf import scf
from pyscf import gto
from pyscf import mcscf

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ["C", (-0.65830719,  0.61123287, -0.00800148)],
    ["C", ( 0.73685281,  0.61123287, -0.00800148)],
    ["H", ( 1.43439081,  1.81898387, -0.00800148)],
    ["H", (-1.35568919,  1.81920887, -0.00868348)],
    ["H", (-1.20806619, -0.34108413, -0.00755148)],
    ["H", ( 1.28636081, -0.34128013, -0.00668648)],]

mol.basis = {'H': 'cc-pvdz',
             'C': 'cc-pvdz',}
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.scf()


class KnowValues(unittest.TestCase):
    def test_casci_4o4e(self):
        mc = mcscf.CASCI(mf, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -77.9734951776, 7)

    def test_casci_6o4e(self):
        mc = mcscf.CASCI(mf, 6, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -77.9746683275, 7)

    def test_casci_6o6e(self):
        mc = mcscf.CASCI(mf, 6, 6)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -77.9804561351, 7)

    def test_mc2step_6o6e(self):
        mc = mcscf.CASSCF(mf, 6, 6)
        mc.conv_tol = 1e-8
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -78.0390051207, 7)

    def test_mc1step_6o6e(self):
        mc = mcscf.CASSCF(mf, 6, 6)
        mc.conv_tol = 1e-8
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -78.0390051207, 7)

    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.conv_tol = 1e-8
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -77.9916207, 6)

    def test_mc1step_4o4e(self):
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.conv_tol = 1e-8
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -77.9916207, 6)


if __name__ == "__main__":
    print("Full Tests for C2H4")
    unittest.main()

