import numpy as np
from pyscf import lib
import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc import gto, scf, ci
from pyscf.pbc.ci import KCIS
import unittest

def setUpModule():
    global cell, kmf, kci, eris
    cell = gto.Cell()
    cell.a = np.eye(3) * 2.5
    cell.mesh = [11] * 3
    cell.atom = '''He    0.    2.       1.5
                   He    1.    1.       1.'''
    cell.basis = {'He': [(0, (1.5, 1)), (0, (1., 1))]}
    cell.build()
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1]), exxdiv=None).run()
    kci = ci.KCIS(kmf)
    eris = kci.ao2mo()

def tearDownModule():
    global cell, kmf, kci, eris
    del cell, kmf, kci, eris

class KnownValues(unittest.TestCase):
    def test_n3_cis_high_cost(self):
        cell = make_test_cell.test_cell_n3(mesh=[29] * 3)
        cell.build()
        kmf_n3_none = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1]), exxdiv=None)
        kmf_n3_none.kernel()
        ekrhf = kmf_n3_none.e_tot
        self.assertAlmostEqual(ekrhf, -8.651923514149, 7)

        # KCIS
        myci = ci.KCIS(kmf_n3_none)
        eris = myci.ao2mo()
        eci, v = myci.kernel(nroots=2, eris=eris, kptlist=[0])
        self.assertAlmostEqual(eci[0][0], 0.223920101177, 5)
        self.assertAlmostEqual(eci[0][1], 0.223920101177, 5)
        eci, v = myci.kernel(nroots=2, eris=eris, kptlist=[1])
        #FIXME: uncertainty for the lowest state?
        self.assertTrue(abs(eci[0][0]-0.291182202333).max() < 1e-5 or
                        abs(eci[0][0]-0.330573456724).max() < 1e-5)
        self.assertAlmostEqual(eci[0][1], 0.330573456724, 5)

    def test_n3_cis_ewald_high_cost(self):
        cell = make_test_cell.test_cell_n3(mesh=[29] * 3)
        cell.build()
        kmf_n3_ewald = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1]), exxdiv='ewald')
        kmf_n3_ewald.kernel()
        self.assertAlmostEqual(kmf_n3_ewald.e_tot, -10.530905169078, 7)

        myci = ci.KCIS(kmf_n3_ewald)
        myci.keep_exxdiv = True
        eris = myci.ao2mo()
        eci, v = myci.kernel(nroots=2, eris=eris, kptlist=[0])
        self.assertAlmostEqual(eci[0][0], 0.693665750383, 5)
        self.assertAlmostEqual(eci[0][1], 0.693665750384, 5)
        eci, v = myci.kernel(nroots=2, eris=eris, kptlist=[1])
        self.assertAlmostEqual(eci[0][0], 0.800318837778, 5)
        self.assertAlmostEqual(eci[0][1], 0.800318837778, 5)

    def test_cis_H(self):
        h = ci.kcis_rhf.cis_H(kci, 0, eris=eris)
        self.assertAlmostEqual(lib.fp(h), 2.979013823936476+0j, 5)
        e0ref, v0ref = np.linalg.eigh(h)

        h = ci.kcis_rhf.cis_H(kci, 1, eris=eris)
        self.assertAlmostEqual(lib.fp(h), 4.046206590499069-0j, 4)
        e1ref, v1ref = np.linalg.eigh(h)

        eci, v = kci.kernel(nroots=3, eris=eris, kptlist=[0, 1])
        self.assertAlmostEqual(abs(e0ref[:3] - eci[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(e1ref[:3] - eci[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(v0ref[:,0].dot(v[0][0])), 1, 6)
        self.assertAlmostEqual(abs(v0ref[:,1].dot(v[0][1])), 1, 6)
        self.assertAlmostEqual(abs(v0ref[:,2].dot(v[0][2])), 1, 6)
        self.assertAlmostEqual(abs(v1ref[:,0].dot(v[1][0])), 1, 6)
        self.assertAlmostEqual(abs(v1ref[:,1].dot(v[1][1])), 1, 6)
        self.assertAlmostEqual(abs(v1ref[:,2].dot(v[1][2])), 1, 6)

    def test_cis_diag(self):
        h = ci.kcis_rhf.cis_H(kci, 1, eris=eris)
        hdiag = kci.get_diag(1, eris=eris)
        self.assertAlmostEqual(abs(h.diagonal() - hdiag).max(), 0, 6)
        self.assertAlmostEqual(lib.fp(hdiag), 5.219997654681162+0j, 5)

    def test_cis_matvec_singlet(self):
        vsize = kci.vector_size()
        vec = np.random.rand(vsize) * 1j
        hc = kci.matvec(vec, 1, eris=eris)
        h = ci.kcis_rhf.cis_H(kci, 1, eris=eris)
        ref = h.dot(vec)
        self.assertAlmostEqual(abs(hc - ref).max(), 0, 7)

    def test_kcis_with_df(self):
        cell = gto.Cell()
        cell.atom = '''
        He 0.000000000000   0.000000000000   0.000000000000
        He 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 7
        cell.output = '/dev/null'
        cell.build()
        kpts = cell.make_kpts([2,1,1])
        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.kernel()
        eci, v = ci.KCIS(kmf).kernel()
        self.assertAlmostEqual(eci[0][0], 1.4249289, 5)
        self.assertAlmostEqual(eci[1][0], 1.5164275, 5)


if __name__ == "__main__":
    print("Full Tests for PBC CIS")
    unittest.main()
