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


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = scf.GHF(mol).get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 14.00554247575052, 9)

    def test_get_veff(self):
        mf = scf.GHF(mol)
        nao = mol.nao_nr()*2
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = mf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 556.53059717681901, 9)

if __name__ == "__main__":
    print("Full Tests for GHF")
    unittest.main()

