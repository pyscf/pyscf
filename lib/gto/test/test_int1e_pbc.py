#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto

mol = gto.Mole()
mol.atom = '''
He     .5    .5      -.5
He    1.     .2       .3
He     .1   -.1       .1 '''
mol.basis = {'He': [(0, (.5, 1)),
                    (1, (.6, 1)),
                    (2, (.8, 1))]}
mol.build()


class KnowValues(unittest.TestCase):
    def test_cint1e_r2_origi(self):
        ref = mol.intor('cint1e_r2_origi_sph')
        dat = mol.intor('cint1e_pbc_r2_origi_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint1e_r4_origi(self):
        ref = mol.intor('cint1e_r4_origi_sph')
        dat = mol.intor('cint1e_pbc_r4_origi_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint3c1e_r2_origk(self):
        ref = mol.intor('cint3c1e_r2_origk_sph')
        dat = mol.intor('cint3c1e_pbc_r2_origk_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint3c1e_r4_origk(self):
        ref = mol.intor('cint3c1e_r4_origk_sph')
        dat = mol.intor('cint3c1e_pbc_r4_origk_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint3c1e_r6_origk(self):
        ref = mol.intor('cint3c1e_r6_origk_sph')
        dat = mol.intor('cint3c1e_pbc_r6_origk_sph')
        self.assertTrue(numpy.allclose(ref, dat))

if __name__ == '__main__':
    print('Full Tests for int1e_pbc')
    unittest.main()
