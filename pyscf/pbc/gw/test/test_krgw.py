#!/usr/bin/env python

import unittest
import numpy as np
from pyscf.pbc import gto, dft, df, scf
from pyscf.pbc.gw import krgw_ac, krgw_cd


def setUpModule():
    global cell, kpts, gdf
    cell = gto.Cell()
    cell.build(
        a = '''
            0.000000     1.783500     1.783500
            1.783500     0.000000     1.783500
            1.783500     1.783500     0.000000
        ''',
        atom = 'C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        verbose = 7,
        output = '/dev/null',
        pseudo = 'gth-pade',
        basis='gth-szv',
        precision=1e-8)

    kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)

def tearDownModule():
    global cell, kpts, gdf
    cell.stdout.close()
    del cell, kpts, gdf

class KnownValues(unittest.TestCase):
    def test_gwcd_high_cost(self):
        kmf = dft.KRKS(cell, kpts).density_fit(with_df=gdf)
        kmf.xc = 'pbe'
        kmf.kernel()

        gw = krgw_cd.KRGWCD(kmf)
        gw.linearized = False

        # without finite size corrections
        gw.fc = False
        nocc = gw.nocc
        gw.kernel(kptlist=[0,1,2],orbs=range(0,nocc+3))
        self.assertAlmostEqual(gw.mo_energy[0][nocc-1], 0.62045796, 4)
        self.assertAlmostEqual(gw.mo_energy[0][nocc],   0.96574426, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc-1], 0.52639129, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc],   1.07442235, 4)

        # with finite size corrections
        gw.fc = True
        gw.kernel(kptlist=[0,1,2],orbs=range(0,nocc+3))
        self.assertAlmostEqual(gw.mo_energy[0][nocc-1], 0.5427707 , 4)
        self.assertAlmostEqual(gw.mo_energy[0][nocc],   0.80148557, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc-1], 0.45073751, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc],   0.92910117, 4)


    def test_finite_size_frequency_convergence(self):
        cell = gto.M(atom='H 0 0 0; H 0 0 .74', a=np.eye(3)*4,
                     basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                     precision=1e-7, verbose=0)
        mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None)
        mf = mf.density_fit(auxbasis='weigend').run(conv_tol=1e-11)
        low = krgw_cd.KRGWCD(mf)
        low.fc = True
        low.kernel(nw=32)
        high = krgw_cd.KRGWCD(mf)
        high.fc = True
        high.kernel(nw=64)
        self.assertTrue(low.converged and high.converged)
        np.testing.assert_allclose(low.mo_energy, high.mo_energy, atol=1e-7)
        self.assertGreater(high.mo_energy[0, 1], high.mo_energy[0, 0])
        selected = krgw_cd.KRGWCD(mf)
        selected.fc = True
        selected.kernel(nw=32, orbs=[1], kptlist=[0])
        self.assertAlmostEqual(selected.mo_energy[0, 1], low.mo_energy[0, 1], places=8)

    def test_contour_deformation_matches_analytic_continuation(self):
        cell = gto.M(atom='H 0 0 0; H 0 0 .74', a=np.eye(3)*4,
                     basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                     precision=1e-6, verbose=0)
        mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None)
        mf = mf.density_fit(auxbasis='weigend').run(conv_tol=1e-10)
        ref = krgw_ac.KRGWAC(mf)
        ref.fc = False
        ref.nw = 32
        ref.ac_pade_npts = 12
        ref.kernel()
        self.assertTrue(np.isfinite(ref.mo_energy).all())
        cd = krgw_cd.KRGWCD(mf)
        cd.fc = False
        cd.kernel(nw=32)
        np.testing.assert_allclose(cd.mo_energy, ref.mo_energy, atol=2e-6)

if __name__ == '__main__':
    print('Full Tests for KRGW')
    unittest.main()
