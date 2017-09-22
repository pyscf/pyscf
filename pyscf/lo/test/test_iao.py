#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf.lo import iao

mol = gto.Mole()
mol.atom = '''
     O    0.   0.       0
     h    0.   -0.757   0.587
     h    0.   0.757    0.587'''
mol.basis = 'unc-sto3g'
mol.verbose = 5
mol.output = '/dev/null'
mol.build()

class KnownValues(unittest.TestCase):
    def test_fast_iao_mulliken_pop(self):
        mf = scf.RHF(mol).run()
        a = iao.iao(mol, mf.mo_coeff[:,mf.mo_occ>0])
        p,chg = iao.fast_iao_mullikan_pop(mol, mf.make_rdm1(), a)
        self.assertAlmostEqual(lib.finger(p), 0.56795867043723325, 5)

        mf = scf.UHF(mol).run()
        p,chg = iao.fast_iao_mullikan_pop(mol, mf.make_rdm1(), a)
        self.assertAlmostEqual(lib.finger(p[0]+p[1]), 0.56795867043723325, 5)


if __name__ == "__main__":
    print("TODO: Test iao")
    unittest.main()



