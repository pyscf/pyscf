#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
import unittest
import numpy
from pyscf import gto, scf, dft
from pyscf.tddft import rks

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['H' , (0. , 0. , .917)],
    ['F' , (0. , 0. , 0.)], ]
mol.basis = '631g'
mol.build()

def finger(a):
    w = numpy.cos(numpy.arange(len(a)))
    return numpy.dot(w, a)

class KnownValues(unittest.TestCase):
    def test_nohbrid_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda, vwn_rpa'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFTNoHybrid(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.059050077236151, 6)

    def test_nohbrid_b88p86(self):
        mf = dft.RKS(mol)
        mf.xc = 'b88,p86'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFTNoHybrid(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -40.462005239920558, 6)

    def test_tddft_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda, vwn_rpa'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.059050077236151, 6)

    def test_tddft_b88p86(self):
        mf = dft.RKS(mol)
        mf.xc = 'b88,p86'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -40.462005239920558, 6)

    def test_tddft_b3pw91(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3pw91'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.218912874291014, 6)

    def test_tddft_b3lyp(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.29609453661341, 6)

    def test_tda_b3lypg(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lypg'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.385520327568869, 6)

    def test_tda_b3pw91(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3pw91'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.313632163628363, 6)

    def test_tda_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.201828219760415, 6)

#NOTE b3lyp by libxc is quite different to b3lyp from xcfun
    def test_tddft_b3lyp_xcfun(self):
        dft.numint._NumInt.libxc = dft.xcfun
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        dft.numint._NumInt.libxc = dft.libxc
        self.assertAlmostEqual(finger(es), finger([9.88975514, 9.88975514, 15.16643994, 30.55289462, 30.55289462]), 6)

    def test_tddft_b3lyp_xcfun(self):
        dft.numint._NumInt.libxc = dft.xcfun
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        dft.numint._NumInt.libxc = dft.libxc
        self.assertAlmostEqual(finger(es), -41.393122257109056, 6)

    def test_tda_lda_xcfun(self):
        dft.numint._NumInt.libxc = dft.xcfun
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        dft.numint._NumInt.libxc = dft.libxc
        self.assertAlmostEqual(finger(es), -41.201828219760415, 6)

    def test_tda_b3lyp_triplet(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.singlet = False
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -40.020204585289648, 6)

    def test_tda_lda_triplet(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.singlet = False
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -39.988118769202416, 6)


if __name__ == "__main__":
    print("Full Tests for TD-RKS")
    unittest.main()
