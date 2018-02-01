import unittest
from pyscf import scf
from pyscf import gto
from pyscf.cc import ccsd
from pyscf.cc import addons

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.verbose = 5
mol.output = '/dev/null'
mol.basis = '631g'
mol.spin = 0
mol.build()
mf1 = scf.RHF(mol).run(conv_tol=1e-12)
gmf = scf.addons.convert_to_ghf(mf1)
myrcc = ccsd.CCSD(mf1).run()

class KnownValues(unittest.TestCase):
    def test_spin2spatial(self):
        t1g = addons.spatial2spin(myrcc.t1)
        t2g = addons.spatial2spin(myrcc.t2)
        orbspin = gmf.mo_coeff.orbspin
        t1a, t1b = addons.spin2spatial(t1g, orbspin)
        t2aa, t2ab, t2bb = addons.spin2spatial(t2g, orbspin)
        self.assertAlmostEqual(abs(myrcc.t1 - t1a).max(), 0, 12)
        self.assertAlmostEqual(abs(myrcc.t2 - t2ab).max(), 0, 12)

        self.assertAlmostEqual(abs(t1g - addons.spatial2spin((t1a,t1b), orbspin)).max(), 0, 12)
        self.assertAlmostEqual(abs(t2g - addons.spatial2spin((t2aa,t2ab,t2bb), orbspin)).max(), 0, 12)

    def test_convert_to_uccsd(self):
        myucc = addons.convert_to_uccsd(myrcc)
        myucc = addons.convert_to_uccsd(myucc)

    def test_convert_to_gccsd(self):
        mygcc = addons.convert_to_uccsd(myrcc)
        mygcc = addons.convert_to_gccsd(myrcc)

        myucc = addons.convert_to_uccsd(myrcc)
        mygcc = addons.convert_to_gccsd(myucc)


if __name__ == "__main__":
    print("Tests for addons")
    unittest.main()


