import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc import gto, scf, ci
from pyscf.pbc.ci import KCIS
import unittest

def setUpModule():
    global cell
    cell = make_test_cell.test_cell_n3(mesh=[29] * 3)
    cell.build()

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_n3_cis_high_cost(self):
        kmf_n3_none = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1]), exxdiv=None)
        kmf_n3_none.kernel()
        ekrhf = kmf_n3_none.e_tot
        self.assertAlmostEqual(ekrhf, -8.651923514149, 6)

        # KCIS
        myci = ci.KCIS(kmf_n3_none)
        eci, v = myci.kernel(nroots=2, kptlist=[0])
        self.assertAlmostEqual(eci[0][0], 0.223920101177, 5)
        self.assertAlmostEqual(eci[0][1], 0.223920101177, 5)
        eci, v = myci.kernel(nroots=2, kptlist=[1])
        #FIXME: value changed around commit de99aaad3 or earliear
        self.assertAlmostEqual(eci[0][0], 0.291182202333, 5)
        self.assertAlmostEqual(eci[0][0], 0.330573456724, 5)
        self.assertAlmostEqual(eci[0][1], 0.330573456724, 5)

    def test_n3_cis_ewald(self):
        kmf_n3_ewald = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1]), exxdiv='ewald')
        kmf_n3_ewald.kernel()
        ekrhf = kmf_n3_ewald.e_tot
        self.assertAlmostEqual(ekrhf, -10.530905169078, 6)

        myci = ci.KCIS(kmf_n3_ewald)
        myci.keep_exxdiv = True
        eci, v = myci.kernel(nroots=2, kptlist=[0])
        self.assertAlmostEqual(eci[0][0], 0.693665750383, 5)
        self.assertAlmostEqual(eci[0][1], 0.693665750384, 5)
        eci, v = myci.kernel(nroots=2, kptlist=[1])
        self.assertAlmostEqual(eci[0][0], 0.760927568875, 5)
        self.assertAlmostEqual(eci[0][1], 0.800318837778, 5)


if __name__ == "__main__":
    print("Full Tests for PBC CIS")
    unittest.main()
