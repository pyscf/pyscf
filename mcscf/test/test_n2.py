#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf

b = 1.4
mol = gto.Mole()
mol.build(
verbose = 5,
output = '/dev/null',
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
)
m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()

molsym = gto.Mole()
molsym.build(
verbose = 5,
output = '/dev/null',
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
symmetry = True
)
msym = scf.RHF(molsym)
msym.conv_tol = 1e-9
msym.scf()


class KnowValues(unittest.TestCase):
    def test_mc1step_4o4e(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 4.17096333, 4)

    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 4.17096333, 4)

    def test_mc1step_6o6e(self):
        mc = mcscf.CASSCF(mol, m, 6, 6)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_6o6e(self):
        mc = mcscf.CASSCF(mol, m, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc1step_symm_4o4e(self):
        mc = mcscf.CASSCF(molsym, msym, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 4.17096333, 4)

    def test_mc2step_symm_4o4e(self):
        mc = mcscf.CASSCF(molsym, msym, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 4.17096333, 4)

    def test_mc1step_symm_6o6e(self):
        mc = mcscf.CASSCF(molsym, msym, 6, 6)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_symm_6o6e(self):
        mc = mcscf.CASSCF(molsym, msym, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_casci_4o4e(self):
        mc = mcscf.CASCI(mol, m, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 4.17125121, 4)

    def test_casci_symm_4o4e(self):
        mc = mcscf.CASCI(molsym, msym, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 4.17125121, 4)

    def test_casci_uhf(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASCI(mol, mf, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 0, 7)

    def test_h1e_for_cas(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        mo = m.mo_coeff
        h0 = mcscf.casci.h1e_for_cas(mc, mo, 4, 5)[0]
        h1 = mcscf.mc1step.h1e_for_cas(mc, mo, mc.update_ao2mo(mo))
        self.assertTrue(numpy.allclose(h0, h1))

    def test_casci_uhf(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASSCF(mol, mf, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc1step_natorb(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        mc.natorb = True
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc2step_natorb(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        mc.natorb = True
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc1step_uhf_natorb(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASSCF(mol, mf, 4, 4)
        mc.natorb = True
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc2step_uhf_natorb(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASSCF(mol, mf, 4, 4)
        mc.natorb = True
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()

