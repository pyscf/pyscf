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
    def test_sfx2c1e(self):
        myx2c = scf.x2c.sfx2c1e(scf.RHF(mol))
        myx2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.081765438081675, 9)

        myx2c.xuncontract = True
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075946103708105, 9)

    def test_x2c1e(self):
        myx2c = scf.x2c.UHF(mol)
        myx2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.081767969229489, 9)

        myx2c.xuncontract = True
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075948634842121, 9)


if __name__ == "__main__":
    print("Full Tests for dhf")
    unittest.main()


