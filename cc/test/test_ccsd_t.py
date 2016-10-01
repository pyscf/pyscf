#!/usr/bin/env python
import unittest
import numpy
from pyscf import gto, scf
from pyscf import cc
from pyscf.cc import ccsd_t


class KnowValues(unittest.TestCase):
    def test_ccsd_t(self):
        mol = gto.M()
        numpy.random.seed(12)
        nocc, nvir = 5, 12
        eris = lambda :None
        eris.ovvv = numpy.random.random((nocc*nvir,nvir*(nvir+1)//2)) * .1
        eris.ovoo = numpy.random.random((nocc,nvir,nocc,nocc)) * .1
        eris.ovov = numpy.random.random((nocc*nvir,nocc*nvir)) * .1
        t1 = numpy.random.random((nocc,nvir)) * .1
        t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
        t2 = t2 + t2.transpose(1,0,3,2)
        mf = scf.RHF(mol)
        mcc = cc.CCSD(mf)
        mcc._scf.mo_energy = numpy.arange(0., nocc+nvir)
        print(ccsd_t.kernel(mcc, eris, t1, t2) + 8.4953387936460398)

if __name__ == "__main__":
    print("Full Tests for CCSD(T)")
    unittest.main()

