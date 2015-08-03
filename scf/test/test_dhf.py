#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import gto
from pyscf import scf

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

mf = scf.dhf.UHF(mol)
mf.conv_tol_grad = 1e-5
mf.scf()


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = mf.init_guess_by_minao()
        self.assertAlmostEqual(abs(dm).sum(), 14.899439258242364, 9)

    def test_get_hcore(self):
        h = mf.get_hcore()
        self.assertAlmostEqual(numpy.linalg.norm(h), 159.55593668675903, 9)

    def test_get_ovlp(self):
        s = mf.get_ovlp()
        self.assertAlmostEqual(numpy.linalg.norm(s), 9.0156256929936056, 9)

    def test_1e(self):
        mf = scf.dhf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.892132873081664, 9)

#    def test_analyze(self):
#        numpy.random.seed(1)
#        pop, chg = mf.analyze()
#        self.assertAlmostEqual(numpy.linalg.norm(pop), 2.0355530265140636, 9)

    def test_scf(self):
        self.assertAlmostEqual(mf.hf_energy, -76.081567943868265, 9)

    def test_rhf(self):
        mf = scf.dhf.RHF(mol)
        mf.conv_tol_grad = 1e-5
        self.assertAlmostEqual(mf.scf(), -76.081567943868265, 9)

    def test_get_veff(self):
        dm = mf.make_rdm1()
        v = mf.get_veff(mol, dm)
        self.assertAlmostEqual(numpy.linalg.norm(v), 56.050212738602092, 9)


if __name__ == "__main__":
    print("Full Tests for dhf")
    unittest.main()

