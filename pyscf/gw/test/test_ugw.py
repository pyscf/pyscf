#!/usr/bin/env python

import unittest
import numpy
from pyscf import lib, gto, scf, dft
from pyscf.gw import ugw_ac
from pyscf.gw import urpa

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = 'O'
    mol.basis = 'aug-cc-pvdz'
    mol.spin = 2
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_gwac_pade(self):
        nocca = (mol.nelectron + mol.spin)//2
        noccb = mol.nelectron - nocca
        gw_obj = ugw_ac.UGWAC(mf, frozen=0)
        gw_obj.linearized = False
        gw_obj.ac = 'pade'
        gw_obj.kernel(orbs=range(nocca-3, nocca+3))
        self.assertAlmostEqual(gw_obj.mo_energy[0][nocca-1], -0.521932084529, 5)
        self.assertAlmostEqual(gw_obj.mo_energy[0][nocca],    0.167547592784, 5)
        self.assertAlmostEqual(gw_obj.mo_energy[1][noccb-1], -0.464605523684, 5)
        self.assertAlmostEqual(gw_obj.mo_energy[1][noccb],   -0.0133557793765, 5)

    def test_rpa(self):
        rpa_obj = urpa.URPA(mf, frozen=0)
        rpa_obj.kernel()

        self.assertAlmostEqual(rpa_obj.e_tot, -74.98369614250653, 6)
        self.assertAlmostEqual(rpa_obj.e_corr, -0.1882153685614803, 6)


if __name__ == "__main__":
    print("Full Tests for UGW")
    unittest.main()
