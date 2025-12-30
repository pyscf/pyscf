#!/usr/bin/env python
import unittest
import numpy as np
from pyscf import gto, scf
from pyscf.mcscf import casci_dft, addons_fomo

class TestCASCI_DFT_Grad(unittest.TestCase):
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
        cls.ncore = 2
    
    def test_dft_casci_gradient(self):
        """Test DFT-CASCI gradient calculation."""
        mc = casci_dft.CASCI(self.mf, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        g = mc.Gradients().kernel()
        self.assertEqual(g.shape, (self.mol.natm, 3))
    
    def test_dft_casci_gradient_vs_numerical(self):
        """Verify DFT-CASCI gradient against numerical differentiation."""
        mc = casci_dft.CASCI(self.mf, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        g_anal = mc.Gradients().kernel()
        
        # Numerical gradient
        step = 1e-4
        coords = self.mol.atom_coords().copy()
        g_num = np.zeros_like(g_anal)
        
        for ia in range(self.mol.natm):
            for ix in range(3):
                coords_p = coords.copy()
                coords_m = coords.copy()
                coords_p[ia, ix] += step
                coords_m[ia, ix] -= step
                
                mol_p = self.mol.set_geom_(coords_p, unit='Bohr', inplace=False)
                mol_m = self.mol.set_geom_(coords_m, unit='Bohr', inplace=False)
                
                mf_p = scf.RHF(mol_p).run()
                mf_m = scf.RHF(mol_m).run()
                
                mc_p = casci_dft.CASCI(mf_p, self.ncas, self.nelecas, xc='LDA')
                mc_m = casci_dft.CASCI(mf_m, self.ncas, self.nelecas, xc='LDA')
                mc_p.kernel()
                mc_m.kernel()
                
                g_num[ia, ix] = (mc_p.e_tot - mc_m.e_tot) / (2 * step)
        
        np.testing.assert_allclose(g_anal, g_num, atol=1e-5)
    
    def test_fomo_casci_dft_gradient(self):
        """Test FOMO-CASCI-DFT gradient calculation."""
        mf_fomo = addons_fomo.fomo_scf(
            self.mf, temperature=0.25, method='gaussian',
            restricted=(self.ncore, self.ncas)
        )
        mf_fomo.kernel()
        
        mc = casci_dft.CASCI(mf_fomo, self.ncas, self.nelecas, xc='PBE')
        mc.kernel()
        g = mc.Gradients().kernel()
        self.assertEqual(g.shape, (self.mol.natm, 3))

if __name__ == '__main__':
    unittest.main()
