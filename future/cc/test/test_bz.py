#!/usr/bin/env python
import unittest
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import cc

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_bz"

mol.atom.extend([
    ["C", (-0.65830719,  0.61123287, -0.00800148)],
    ["C", ( 0.73685281,  0.61123287, -0.00800148)],
    ["C", ( 1.43439081,  1.81898387, -0.00800148)],
    ["C", ( 0.73673681,  3.02749287, -0.00920048)],
    ["C", (-0.65808819,  3.02741487, -0.00967948)],
    ["C", (-1.35568919,  1.81920887, -0.00868348)],
    ["H", (-1.20806619, -0.34108413, -0.00755148)],
    ["H", ( 1.28636081, -0.34128013, -0.00668648)],
    ["H", ( 2.53407081,  1.81906387, -0.00736748)],
    ["H", ( 1.28693681,  3.97963587, -0.00925948)],
    ["H", (-1.20821019,  3.97969587, -0.01063248)],
    ["H", (-2.45529319,  1.81939187, -0.00886348)],])


mol.basis = {"H": '6-31g',
             "C": '6-31g',}
mol.build()

rhf = scf.RHF(mol)
rhf.scf()


class KnowValues(unittest.TestCase):
    def test_ccsd(self):
        mcc = cc.ccsd.CC(mol, rhf)
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        mcc.ccsd()
        self.assertTrue(numpy.allclose(mcc.t2,mcc.t2.transpose(1,0,3,2)))
        self.assertAlmostEqual(mcc.ecc, -0.5690403273511450, 8)
        self.assertAlmostEqual(abs(mcc.t2).sum(), 92.61277290776878, 6)


if __name__ == "__main__":
    print "Full Tests for C6H6"
    unittest.main()



