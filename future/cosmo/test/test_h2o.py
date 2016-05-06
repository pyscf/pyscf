import unittest
from pyscf import gto, scf, dft, mcscf
from pyscf import cosmo

mol = gto.Mole()
mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
               H                 -0.00000000   -0.84695236    0.59109389
               H                 -0.00000000    0.89830571    0.52404783 '''
mol.basis = 'cc-pvdz'
mol.output = '/dev/null'
mol.build()

class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        mf = cosmo.cosmo_(scf.RHF(mol))
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.kernel(), -76.0030675691, 9)

    def test_nr_rks(self):
        mf = cosmo.cosmo_(dft.RKS(mol))
        mf.xc = 'b3lypg'
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.kernel(), -76.407553915, 9)

    def test_nr_CASSCF(self):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        mc = cosmo.cosmo_(mcscf.CASSCF(mf, 4, 4))
        mc.verbose = 4
        self.assertAlmostEqual(mc.kernel(mc.sort_mo([3,4,6,7]))[0],
                               -76.0656799183, 8)

    def test_nr_CASCI(self):
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        mc = cosmo.cosmo_(mcscf.CASCI(mf, 4, 4))
        self.assertAlmostEqual(mc.kernel(mc.sort_mo([3,4,6,7]))[0],
                               -76.0107164465, 8)

if __name__ == "__main__":
    print("Full Tests for COSMO")
    unittest.main()
