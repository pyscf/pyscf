#!/usr/bin/env python
import unittest
import numpy
import copy
from pyscf import gto, scf, lib, symm
from pyscf import cc
from pyscf.cc import uccsd_t

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -.757 , .587)],
    [1 , (0. ,  .757 , .587)]]
mol.spin = 2
mol.basis = '3-21g'
mol.symmetry = 'C2v'
mol.build()
mol1 = copy.copy(mol)
mol1.symmetry = False

mf = scf.UHF(mol1).run(conv_tol=1e-14)
mcc = cc.UCCSD(mf)
mcc.conv_tol = 1e-14
mcc.kernel()

class KnownValues(unittest.TestCase):
    def test_uccsd_t(self):
        mf1 = copy.copy(mf)
        nao, nmo = mf.mo_coeff[0].shape
        numpy.random.seed(10)
        mf1.mo_coeff = numpy.random.random((2,nao,nmo))
        numpy.random.seed(12)
        nocca, noccb = mol.nelec
        nmo = mf1.mo_occ[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        t1a  = .1 * numpy.random.random((nocca,nvira))
        t1b  = .1 * numpy.random.random((noccb,nvirb))
        t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
        t2aa = t2aa - t2aa.transpose(0,1,3,2)
        t2aa = t2aa - t2aa.transpose(1,0,2,3)
        t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
        t2bb = t2bb - t2bb.transpose(0,1,3,2)
        t2bb = t2bb - t2bb.transpose(1,0,2,3)
        t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
        t1 = t1a, t1b
        t2 = t2aa, t2ab, t2bb
        mycc = cc.UCCSD(mf1)
        eris = mycc.ao2mo(mf1.mo_coeff)
        e3a = uccsd_t.kernel(mycc, eris, [t1a,t1b], [t2aa, t2ab, t2bb])
        self.assertAlmostEqual(e3a, 9877.2780859693339, 6)

        mycc.max_memory = 0
        e3a = uccsd_t.kernel(mycc, eris, [t1a,t1b], [t2aa, t2ab, t2bb])
        self.assertAlmostEqual(e3a, 9877.2780859693339, 6)

        e3a = mcc.ccsd_t()
        self.assertAlmostEqual(e3a, -0.0009857042572475674, 11)

    #def test_uccsd_t_symm(self):
    #    mf = scf.UHF(mol).run(conv_tol=1e-14)
    #    mcc = cc.UCCSD(mf)
    #    mcc.conv_tol = 1e-14
    #    e3a = mcc.run().ccsd_t()
    #    self.assertAlmostEqual(e3a, -0.0030600226107389866, 11)


if __name__ == "__main__":
    print("Full Tests for UCCSD(T)")
    unittest.main()

