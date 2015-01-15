#!/usr/bin/env python

import unittest
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.verbose = 0
mol.output = '/dev/null'
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol.basis = {"H": '6-31g',
             "O": '6-31g',}
mol.build()


class KnowValues(unittest.TestCase):
    def test_project_mo_nr2nr(self):
        nao = mol.nao_nr()
        c = numpy.random.random((nao,nao))
        c1 = scf.addons.project_mo_nr2nr(mol, c, mol)
        self.assertTrue(numpy.allclose(c, c1))

        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_nr2nr(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 108.7133838914646, 12)

    def test_project_mo_r2r(self):
        nao = mol.nao_2c()
        c = numpy.random.random((nao*2,nao*2))
        c = c + numpy.sin(c)*1j
        c1 = scf.addons.project_mo_r2r(mol, c, mol)
        self.assertTrue(numpy.allclose(c, c1))

        numpy.random.seed(15)
        n2c = mol.nao_2c()
        n4c = n2c * 2
        mo1 = numpy.random.random((n4c,n4c)) + numpy.random.random((n4c,n4c))*1j
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_r2r(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 2825.412778091546, 12)

    def test_project_mo_nr2r(self):
        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_nr2r(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 224.6914959998213, 12)

    def test_frac_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf.get_occ = scf.addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -107.35665427207526, 9)

    def test_dynamic_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            6      0.   0  -0.7
            6      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf.get_occ = scf.addons.dynamic_occ(mf)
        self.assertAlmostEqual(mf.scf(), -74.214503776693817, 9)

    def test_follow_state(self):
        mf = scf.RHF(mol)
        mf.scf()
        mo0 = mf.mo_coeff[:,[0,1,2,3,5]]
        mf.get_occ = scf.addons.follow_state(mf, mo0)
        self.assertAlmostEqual(mf.scf(), -75.178145727548511, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[:6], [2,2,2,2,0,2]))

    def test_symm_allow_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf.get_occ = scf.addons.symm_allow_occ(mf)
        self.assertAlmostEqual(mf.scf(), -106.49900188208861, 9)

    def test_float_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            C      0.   0   0'''
        mol.basis = 'cc-pvdz'
        mol.build()
        mf = scf.UHF(mol)
        mf.get_occ = scf.addons.float_occ(mf)
        self.assertAlmostEqual(mf.scf(), -37.590712883365917, 9)


if __name__ == "__main__":
    print "Full Tests for addons"
    unittest.main()

