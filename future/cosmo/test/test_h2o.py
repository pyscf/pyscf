import unittest
from pyscf import gto, scf, dft
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
        mf = cosmo.icosmo.cosmo_for_rhf(scf.RHF(mol))
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.kernel(), -76.0030469182, 9)

    def test_nr_rks(self):
        mf = cosmo.icosmo.cosmo_for_rhf(dft.RKS(mol))
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.kernel(), -76.4076073815, 9)

if __name__ == "__main__":
    print("Full Tests for COSMO")
    unittest.main()
