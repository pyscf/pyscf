#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fciqmcscf

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
    def test_mc2step_4o4e_fci(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        emc = mc.mc2step()[0] + mol.nuclear_repulsion()
        self.assertAlmostEqual(emc,-108.91378640707609, 7)

    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(mol, m, 4, 4)
        mc.max_cycle_macro = 10
        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
        mc.fcisolver.RDMSamples = 5000
        emc = mc.mc2step()[0] + mol.nuclear_repulsion()
        self.assertAlmostEqual(emc,-108.91378666934476, 7)

    def test_mc2step_6o6e(self):
        mc = mcscf.CASSCF(mol, m, 6, 6)
        mc.max_cycle_macro = 10
        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
        mc.fcisolver.RDMSamples = 5000
        emc = mc.mc2step()[0] + mol.nuclear_repulsion()
        self.assertAlmostEqual(emc,-108.98028859357791, 7)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()

