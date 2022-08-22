#!/usr/bin/env python

import unittest
import numpy
import os
from pyscf import lib
from pyscf.pbc import gto, dft, scf, df
from pyscf.pbc.gw import kugw_ac

def setUpModule():
    global cell, kmf, kpts
    cell = gto.Cell()
    cell.build(
        unit = 'B',
        a = [[ 0.,          6.74027466,  6.74027466],
             [ 6.74027466,  0.,          6.74027466],
             [ 6.74027466,  6.74027466,  0.        ]],
        atom = '''H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733''',
        basis = 'gth-dzvp',
        pseudo = 'gth-pade',
        verbose = 7,
        output = '/dev/null',
        charge = 0,
        spin = None)
    cell.spin = 3
    kpts = cell.make_kpts([3,1,1], scaled_center=[0,0,0])
    kmf = scf.KUHF(cell, kpts, exxdiv=None).density_fit()
    kmf.run()

def tearDownModule():
    global cell, kmf
    cell.stdout.close()
    del cell, kmf

class KnownValues(unittest.TestCase):
    def test_gwac_pade(self):
        gw = kugw_ac.KUGWAC(kmf)
        gw.linearized = False
        gw.ac = 'pade'
        gw.fc = False
        nocca, noccb = gw.nocc
        gw.kernel(kptlist=[0,1,2], orbs=range(0, nocca+3))
        self.assertAlmostEqual(gw.mo_energy[0][0][nocca-1], -0.28012813, 5)
        self.assertAlmostEqual(gw.mo_energy[0][0][nocca],    0.13748876, 5)
        self.assertAlmostEqual(gw.mo_energy[0][1][nocca-1], -0.29515851, 5)
        self.assertAlmostEqual(gw.mo_energy[0][1][nocca],    0.14128011, 5)
        self.assertAlmostEqual(gw.mo_energy[1][0][noccb-1], -0.33991721, 5)
        self.assertAlmostEqual(gw.mo_energy[1][0][noccb],    0.10578847, 5)
        self.assertAlmostEqual(gw.mo_energy[1][1][noccb-1], -0.33547973, 5)
        self.assertAlmostEqual(gw.mo_energy[1][1][noccb],    0.08053408, 5)

        gw.fc = True
        nocca, noccb = gw.nocc
        gw.kernel(kptlist=[0,1,2], orbs=range(0,nocca+3))
        self.assertAlmostEqual(gw.mo_energy[0][0][nocca-1], -0.40244058, 5)
        self.assertAlmostEqual(gw.mo_energy[0][0][nocca],    0.13618348, 5)
        self.assertAlmostEqual(gw.mo_energy[0][1][nocca-1], -0.41743063, 5)
        self.assertAlmostEqual(gw.mo_energy[0][1][nocca],    0.13997427, 5)
        self.assertAlmostEqual(gw.mo_energy[1][0][noccb-1], -0.46133481, 5)
        self.assertAlmostEqual(gw.mo_energy[1][0][noccb],    0.1044926 , 5)
        self.assertAlmostEqual(gw.mo_energy[1][1][noccb-1], -0.4568894 , 5)
        self.assertAlmostEqual(gw.mo_energy[1][1][noccb],    0.07922511, 5)

if __name__ == '__main__':
    print('Full Tests for KUGW')
    unittest.main()
