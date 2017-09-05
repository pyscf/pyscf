#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf.prop import nmr

mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'

mol.atom = [
    [1   , (0. , 0. , .917)],
    ["F" , (0. , 0. , 0.)], ]
#mol.nucmod = {"F":2, "H":2}
mol.basis = {"H": 'cc_pvdz',
             "F": 'cc_pvdz',}
mol.build()

nrhf = scf.RHF(mol)
nrhf.conv_tol_grad = 1e-6
nrhf.conv_tol = 1e-12
nrhf.scf()

rhf = scf.dhf.RHF(mol)
nrhf.conv_tol_grad = 1e-6
nrhf.conv_tol = 1e-12
rhf.scf()

def finger(mat):
    return abs(mat).sum()

class KnowValues(unittest.TestCase):
    def test_nr_common_gauge_ucpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = False
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1636.7413165636394, 5)

    def test_nr_common_gauge_cpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = True
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1562.3859473764551, 5)

    def test_nr_giao_ucpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = False
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1488.0948832784416, 5)

    def test_nr_giao_cpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = True
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1358.9826064972372, 5)

    def test_rmb_common_gauge_ucpscf(self):
        m = nmr.DHF(rhf)
        m.cphf = False
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1642.1878944169432, 5)

    def test_rmb_common_gauge_cpscf(self):
        m = nmr.DHF(rhf)
        m.cphf = True
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1569.0431469270311, 5)

    def test_rmb_giao_ucpscf(self):
        m = nmr.DHF(rhf)
        m.cphf = False
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1493.7243596881835, 5)

    def test_rmb_giao_cpscf(self):
        m = nmr.DHF(rhf)
        m.cphf = True
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1365.4713349881783, 5)

    def test_rkb_giao_cpscf(self):
        m = nmr.DHF(rhf)
        m.mb = 'RKB'
        m.cphf = True
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1923.9127246223129, 5)

    def test_rkb_common_gauge_cpscf(self):
        m = nmr.DHF(rhf)
        m.mb = 'RKB'
        m.cphf = True
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1980.1211286797848, 5)

    def test_make_h10(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm0 = numpy.random.random((nao,nao))
        dm0 = dm0 + dm0.T
        h1 = nmr.rhf.make_h10(mol, dm0)
        self.assertAlmostEqual(numpy.linalg.norm(h1), 21.255203821714673, 8)
        h1 = nmr.rhf.make_h10(mol, dm0, gauge_orig=(0,0,0))
        self.assertAlmostEqual(numpy.linalg.norm(h1), 4.020198783142229, 8)
        h1 = nmr.dhf.make_h10(mol, rhf.make_rdm1())
        self.assertAlmostEqual(numpy.linalg.norm(h1), 15.041326056582273, 8)
        h1 = nmr.dhf.make_h10(mol, rhf.make_rdm1(), gauge_orig=(0,0,0), mb='RKB')
        self.assertAlmostEqual(numpy.linalg.norm(h1), 7.3636964305440609, 8)



if __name__ == "__main__":
    print("Full Tests of RHF-MSC DHF-MSC for HF")
    unittest.main()
