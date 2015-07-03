#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf

b = 1.4
mol = gto.M(
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

molsym = gto.M(
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
        mc = mcscf.CASSCF(m, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_4o4e_internal_rotation(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mc.internal_rotation = True
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(m, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_6o6e(self):
        mc = mcscf.CASSCF(m, 6, 6)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_6o6e(self):
        mc = mcscf.CASSCF(m, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc1step_symm_4o4e(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc2step_symm_4o4e(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_symm_6o6e(self):
        mc = mcscf.CASSCF(msym, 6, 6)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_symm_6o6e(self):
        mc = mcscf.CASSCF(msym, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_casci_4o4e(self):
        mc = mcscf.CASCI(m, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910275883606078, 4)

    def test_casci_symm_4o4e(self):
        mc = mcscf.CASCI(msym, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910275883606078, 4)

    def test_casci_uhf(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASCI(mf, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()), 0, 7)

    def test_h1e_for_cas(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mo = m.mo_coeff
        h0 = mcscf.casci.h1e_for_cas(mc, mo, 4, 5)[0]
        h1 = mcscf.mc1step.h1e_for_cas(mc, mo, mc.ao2mo(mo))
        self.assertTrue(numpy.allclose(h0, h1))

    def test_casci_uhf(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASSCF(mf, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_frozen1s(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        mc.frozen = 3
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.91373646206542, 7)

    def test_frozenselect(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        mc.frozen = [i-1 for i in [19, 20, 26, 27]]
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.91238513746941, 7)

    def test_wfnsym(self):
        mc = mcscf.CASSCF(msym, 4, (3,1))
        mc.fcisolver.wfnsym = 4
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.74508322877787, 7)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()

