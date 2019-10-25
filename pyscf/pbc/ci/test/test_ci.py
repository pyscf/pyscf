import pyscf.pbc.cc.test.make_test_cell as make_test_cell
from pyscf.pbc import gto, scf, ci
from pyscf.pbc.ci import KCIS
import unittest

cell = make_test_cell.test_cell_n3()
cell.mesh = [29] * 3
cell.build()
kmf_n3_none = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1]), exxdiv=None)
kmf_n3_none.kernel()
kmf_n3_ewald = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1]), exxdiv='ewald')
kmf_n3_ewald.kernel()

def tearDownModule():
    global cell, kmf_n3_none, kmf_n3_ewald
    del cell, kmf_n3_none, kmf_n3_ewald

class KnownValues(unittest.TestCase):
    def test_n3_cis(self):
        ehf_bench = [-8.651923514149, -10.530905169078]

        ekrhf = kmf_n3_none.e_tot
        self.assertAlmostEqual(ekrhf, ehf_bench[0], 6)
        ekrhf = kmf_n3_ewald.e_tot
        self.assertAlmostEqual(ekrhf, ehf_bench[1], 6)

        # KCIS
        myci = ci.KCIS(kmf_n3_none)
        eci, v = myci.kernel(nroots=2, kptlist=[0])
        self.assertAlmostEqual(eci[0][0], 0.223920101177)
        self.assertAlmostEqual(eci[0][1], 0.223920101177)
        eci, v = myci.kernel(nroots=2, kptlist=[1])
        self.assertAlmostEqual(eci[0][0], 0.291182202333)
        self.assertAlmostEqual(eci[0][1], 0.330573456724)

        myci = ci.KCIS(kmf_n3_ewald)
        myci.keep_exxdiv = True
        eci, v = myci.kernel(nroots=2, kptlist=[0])
        self.assertAlmostEqual(eci[0][0], 0.693665750383)
        self.assertAlmostEqual(eci[0][1], 0.693665750384)
        eci, v = myci.kernel(nroots=2, kptlist=[1])
        self.assertAlmostEqual(eci[0][0], 0.760927568875)
        self.assertAlmostEqual(eci[0][1], 0.800318837778)
        









