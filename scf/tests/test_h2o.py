#
# File: test_h2o.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import copy
import unittest
import scf
import scf.dhf as dhf
import gto

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_h2o"
mol.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

mol.basis = {"H": '6-31g',
             "O": '6-31g',}
mol.build()


class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        rhf.scf_threshold = 1e-11
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_uhf(self):
        uhf = scf.UHF(mol)
        uhf.scf_threshold = 1e-11
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_nr_rhf_no_mem(self):
        rhf = scf.RHF(mol)
        rhf.scf_threshold = 1e-11
        rhf.eri_in_memory = False
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_uhf_no_mem(self):
        uhf = scf.UHF(mol)
        uhf.scf_threshold = 1e-11
        uhf.eri_in_memory = False
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_nr_rhf_no_direct(self):
        rhf = scf.RHF(mol)
        rhf.scf_threshold = 1e-11
        rhf.eri_in_memory = False
        rhf.direct_scf = False
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_uhf_no_direct(self):
        uhf = scf.UHF(mol)
        uhf.scf_threshold = 1e-11
        uhf.eri_in_memory = False
        uhf.direct_scf = False
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

#    def test_r_uhf(self):
#        uhf = dhf.UHF(mol)
#        self.assertAlmostEqual(uhf.scf(), -76.03852219027, 9)
#
#    def test_r_rhf(self):
#        uhf = dhf.RHF(mol)
#        self.assertAlmostEqual(uhf.scf(), -76.03852219016, 9)
#
#    def test_r_dhf_dkb(self):
#        dkb = scf.dhf_dkb.UHF(mol)
#        self.assertAlmostEqual(dkb.scf(), -76.03856379177, 9)
#
#    def test_r_rhf_gaunt(self):
#        uhf = dhf.RHF(mol)
#        uhf.with_gaunt = True
#        self.assertAlmostEqual(uhf.scf(), -76.03070215725, 9)


    def test_level_shift_uhf(self):
        uhf = scf.UHF(mol)
        uhf.level_shift_factor = 1.2
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_nuclear_repulsion(self):
        self.assertAlmostEqual(mol.nuclear_repulsion(), 9.18825841775, 10)

    def test_nr_rhf_symm(self):
        mol1 = mol.copy()
        mol1.symmetry = 1
        mol1.build()
        rhf = scf.hf.RHF(mol1)
        rhf.scf_threshold = 1e-11
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_uhf_symm(self):
        mol1 = mol.copy()
        mol1.symmetry = 1
        mol1.build()
        uhf = scf.hf.UHF(mol1)
        uhf.scf_threshold = 1e-11
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)


if __name__ == "__main__":
    print "Full Tests for H2O"
    unittest.main()
