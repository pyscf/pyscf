import unittest
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.dft import dks

class KnownValues(unittest.TestCase):
    def test_dks_lda(self):
        mol = gto.Mole()
        mol.atom = [['Ne',(0.,0.,0.)]]
        mol.basis = 'uncsto3g'
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.build()
        mf = dks.DKS(mol)
        mf.xc = 'lda'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -126.041808355268, 9)


if __name__ == "__main__":
    print("Test DKS")
    unittest.main()

