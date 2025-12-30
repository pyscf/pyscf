#!/usr/bin/env python
import unittest
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import casci_dft, addons_fomo

class TestCASCI_DFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g',
            unit='Angstrom'
        )
        cls.mf = scf.RHF(cls.mol).run()
        cls.ncas = 4
        cls.nelecas = 4
    
    def test_dft_casci_energy(self):
        """Test DFT-CASCI energy calculation."""
        mc = casci_dft.CASCI(self.mf, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)
        self.assertLess(mc.e_tot, -74.0)
    
    def test_different_functionals(self):
        """Test DFT-CASCI with different XC functionals."""
        for xc in ['LDA', 'PBE', 'B3LYP']:
            mc = casci_dft.CASCI(self.mf, self.ncas, self.nelecas, xc=xc)
            mc.kernel()
            self.assertIsNotNone(mc.e_tot)
    
    def test_fomo_casci(self):
        """Test FOMO-CASCI energy calculation."""
        ncore = 2
        mf_fomo = addons_fomo.fomo_scf(
            self.mf, temperature=0.25, method='gaussian',
            restricted=(ncore, self.ncas)
        )
        mf_fomo.kernel()
        mc = mcscf.CASCI(mf_fomo, self.ncas, self.nelecas)
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)
    
    def test_fomo_casci_dft(self):
        """Test FOMO-CASCI with DFT core."""
        ncore = 2
        mf_fomo = addons_fomo.fomo_scf(
            self.mf, temperature=0.25, method='gaussian',
            restricted=(ncore, self.ncas)
        )
        mf_fomo.kernel()
        mc = casci_dft.CASCI(mf_fomo, self.ncas, self.nelecas, xc='PBE')
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)

if __name__ == '__main__':
    unittest.main()
