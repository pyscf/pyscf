#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf import nmr

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'

mol.atom = '''
     O      0.   0.       0.
     H      0.  -0.757    0.587
     H      0.   0.757    0.587'''
mol.basis = 'ccpvdz'
mol.build()

def finger(mat):
    w = numpy.cos(numpy.arange(mat.size))
    return numpy.dot(w, mat.ravel())

class KnowValues(unittest.TestCase):
    def test_nr_lda_common_gauge(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.xc = 'lda,vwn'
        mf.scf()
        m = nmr.RKS(mf)
        m.gauge_orig = (1,1,1)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 13.746716596160255, 7)

    def test_nr_b3lyp_common_gauge(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.xc = 'b3lyp'
        mf.scf()
        m = nmr.RKS(mf)
        m.gauge_orig = (1,1,1)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 15.208617349962076, 7)

    def test_nr_lda_giao(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.xc = 'lda,vwn'
        mf.scf()
        m = nmr.RKS(mf)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 58.258284484160477, 7)

    def test_nr_b3lyp_giao(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.xc = 'b3lyp'
        mf.scf()
        m = nmr.RKS(mf)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 54.531980590047993, 7)



if __name__ == "__main__":
    print("Full Tests of RHF-MSC DHF-MSC for HF")
    unittest.main()
    import sys; sys.exit()
