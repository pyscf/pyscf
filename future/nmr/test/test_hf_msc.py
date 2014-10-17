#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import nmr

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_hf"

mol.atom.extend([
    [1   , (0. , 0. , .917)],
    ["F" , (0. , 0. , 0.)], ])
#mol.nucmod = {"F":2, "H":2}
mol.basis = {"H": 'cc_pvdz',
             "F": 'cc_pvdz',}
mol.build()

nrhf = scf.RHF(mol)
nrhf.conv_threshold = 1e-11
nrhf.scf()

rhf = scf.dhf.RHF(mol)
rhf.conv_threshold = 1e-11
rhf.scf()

def finger(mat):
    return abs(mat).sum()

class KnowValues(unittest.TestCase):
    def test_nr_common_gauge_ucpscf(self):
        m = nmr.hf.MSC(nrhf)
        m.is_cpscf = False
        m.gauge_orig = (1,1,1)
        m.is_giao = False
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1636.7415677000859, 8)

    def test_nr_common_gauge_cpscf(self):
        m = nmr.hf.MSC(nrhf)
        m.is_cpscf = True
        m.gauge_orig = (1,1,1)
        m.is_giao = False
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1562.3861566059275, 8)

    def test_nr_giao_ucpscf(self):
        m = nmr.hf.MSC(nrhf)
        m.is_cpscf = False
        m.is_giao = True
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1488.0951043100554, 8)

    def test_nr_giao_cpscf(self):
        m = nmr.hf.MSC(nrhf)
        m.is_cpscf = True
        m.is_giao = True
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1358.9828207216542, 8)

    def test_r_common_gauge_ucpscf(self):
        m = nmr.dhf.MSC(rhf)
        m.is_cpscf = False
        m.gauge_orig = (1,1,1)
        m.is_giao = False
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1642.1875087918286, 8)

    def test_r_common_gauge_cpscf(self):
        m = nmr.dhf.MSC(rhf)
        m.is_cpscf = True
        m.gauge_orig = (1,1,1)
        m.is_giao = False
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1569.0406406569753, 8)

    def test_r_giao_ucpscf(self):
        m = nmr.dhf.MSC(rhf)
        m.is_cpscf = False
        m.is_giao = True
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1493.7232146398119, 8)

    def test_r_giao_cpscf(self):
        m = nmr.dhf.MSC(rhf)
        m.is_cpscf = True
        m.is_giao = True
        msc = m.msc()
        self.assertAlmostEqual(finger(msc), 1365.4686193737755, 8)


if __name__ == "__main__":
    print "Full Tests of RHF-MSC DHF-MSC for HF"
    unittest.main()
