#!/usr/bin/env python

import unittest
import numpy
import os
from pyscf import lib
from pyscf.pbc import gto, dft, scf, df
from pyscf.pbc.gw import krgw_ac
from pyscf.pbc.gw import krgw_cd

def setUpModule():
    global cell
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

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):
    def test_gwac_pade_high_cost(self):
        kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
        gdf = df.GDF(cell, kpts)
        gdf._cderi = gdf_fname = os.path.realpath(os.path.join(__file__, '..', 'gdf_ints_311.h5'))
        if 1:
            gdf._cderi_to_save = gdf_fname
            gdf.build()

        kmf = dft.KRKS(cell, kpts).density_fit(with_df=gdf)
        kmf.xc = 'pbe'
        kmf.kernel()

        gw = krgw_ac.KRGWAC(kmf)
        gw.linearized = False
        gw.ac = 'pade'

        # without finite size corrections
        gw.fc = False
        nocc = gw.nocc
        gw.kernel(kptlist=[0,1,2],orbs=range(0, nocc+3))
        self.assertAlmostEqual(gw.mo_energy[0][nocc-1], 0.62045797, 4)
        self.assertAlmostEqual(gw.mo_energy[0][nocc]  , 0.96574324, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc-1], 0.52639137, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc]  , 1.07513258, 4)

        # with finite size corrections
        gw.fc = True
        gw.kernel(kptlist=[0,1,2], orbs=range(0, nocc+3))
        self.assertAlmostEqual(gw.mo_energy[0][nocc-1], 0.54277092, 4)
        self.assertAlmostEqual(gw.mo_energy[0][nocc]  , 0.80148537, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc-1], 0.45073793, 4)
        self.assertAlmostEqual(gw.mo_energy[1][nocc]  , 0.92910108, 4)

    def test_gwcd_high_cost(self):
        kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
        gdf = df.GDF(cell, kpts)
        gdf._cderi = gdf_fname = os.path.realpath(os.path.join(__file__, '..', 'gdf_ints_311.h5'))
        if 1:
            gdf._cderi_to_save = gdf_fname
            gdf.build()

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

if __name__ == '__main__':
    print('Full Tests for KRGW')
    unittest.main()
