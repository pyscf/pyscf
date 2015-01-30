#!/usr/bin/env python
import unittest
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import cc

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = {'H': 'cc-pvdz',
             'O': 'cc-pvdz',}
mol.build()
rhf = scf.RHF(mol)
rhf.scf()


class KnowValues(unittest.TestCase):
    def test_ccsd(self):
        mcc = cc.ccsd.CC(rhf)
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        eris = mcc.ao2mo()
        emp2, t1, t2 = mcc.init_amps(eris)
        self.assertAlmostEqual(abs(t2).sum(), 4.9556571218177, 12)
        self.assertAlmostEqual(emp2, -0.2040199672883385, 12)
        t1, t2 = cc.ccsd.update_amps(mcc, t1, t2, eris)
        self.assertAlmostEqual(abs(t1).sum(), 0.0475038989126  , 12)
        self.assertAlmostEqual(abs(t2).sum(), 5.401823846018721, 12)
        self.assertAlmostEqual(cc.ccsd.energy(mcc, t1, t2, eris),
                               -0.208967840546667, 12)
        t1, t2 = cc.ccsd.update_amps(mcc, t1, t2, eris)
        self.assertAlmostEqual(cc.ccsd.energy(mcc, t1, t2, eris),
                               -0.212173678670510, 12)
        self.assertAlmostEqual(abs(t1).sum(), 0.05470123093500083, 12)
        self.assertAlmostEqual(abs(t2).sum(), 5.5605208391876539, 12)

        mcc.ccsd()
        self.assertTrue(numpy.allclose(mcc.t2,mcc.t2.transpose(1,0,3,2)))
        self.assertAlmostEqual(mcc.ecc, -0.2133432312951, 8)
        self.assertAlmostEqual(abs(mcc.t2).sum(), 5.63970279799556984, 6)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()

