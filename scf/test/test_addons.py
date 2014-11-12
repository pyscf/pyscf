#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

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


if __name__ == "__main__":
    print "Full Tests for addons"
    unittest.main()

