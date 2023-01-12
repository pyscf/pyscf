#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Matthew Hennefarth <matthew.hennefarth@gmail.com>

import unittest
import numpy as np
from pyscf import gto, scf
import pyscf.md as md

CHECK_STABILITY = False

def setUpModule():
    global h2o, hf_scanner, casscf_scanner
    h2o = gto.M(verbose=3,
                output='/dev/null',
                atom=[['O', 0, 0, 0], ['H', 0, -0.757, 0.587],
                      ['H', 0, 0.757, 0.587]],
                basis='def2-svp')

    hf_scanner = scf.RHF(h2o)
    hf_scanner.build()
    hf_scanner.conv_tol_grad = 1e-6
    hf_scanner.max_cycle = 700

    ethylene = gto.M(verbose=3,
                     output='/dev/null',
                     atom=''' C -0.0110224 -0.01183 -0.0271398
        C -0.000902273 0.0348566 1.34708
        H 1.07646 0.0030022 -0.456854
        H 0.976273 -0.140089 1.93039
        H -0.926855 -0.147441 1.98255
        H -0.983897 0.0103535 -0.538589
        ''',
                     unit='ANG',
                     basis='ccpvdz',
                     spin=0)

    hf = ethylene.RHF().run()
    ncas, nelecas = (2, 2)
    casscf_scanner = hf.CASSCF(ncas, nelecas)
    casscf_scanner.conv_tol = 1e-12
    casscf_scanner.conv_tol_grad = 1e-6


def tearDownModule():
    global h2o, hf_scanner, casscf_scanner
    hf_scanner.stdout.close()
    casscf_scanner.stdout.close()
    del h2o, hf_scanner, casscf_scanner


class KnownValues(unittest.TestCase):

    def test_hf_water_zero_init_veloc(self):
        driver = md.NVE(hf_scanner, dt=10, steps=10)

        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 0.000349066856492198, 9)
        self.assertAlmostEqual(driver.epot, -75.96132729628864, 9)

        final_coord = np.array([[0.0000000000, 0.0000000000, 0.0020720320],
                                [0.0000000000, -1.4113069887, 1.0928269088],
                                [0.0000000000, 1.4113069887, 1.0928269088]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        if CHECK_STABILITY:
            beginning_energy = driver.ekin + driver.epot
            driver.steps = 990
            driver.kernel()
            self.assertAlmostEqual(driver.ekin + driver.epot, beginning_energy,
                                   4)

    def test_hf_water_init_veloc(self):
        init_veloc = np.array([[0.000336, 0.000044, 0.000434],
                               [-0.000364, -0.000179, 0.001179],
                               [-0.001133, -0.000182, 0.000047]])

        driver = md.NVE(hf_scanner, mol=h2o, veloc=init_veloc, dt=5, steps=10)

        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 0.0068732364518669445, 8)
        self.assertAlmostEqual(driver.epot, -75.96078835576954, 8)

        final_coord = np.array([[0.0151120306, 0.0017437807, 0.0201833153],
                                [-0.0163089669, -1.4306397410, 1.1556586744],
                                [-0.0509295530, 1.4181437135, 1.1076812077]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        if CHECK_STABILITY:
            beginning_energy = driver.ekin + driver.epot

            driver.steps = 990
            driver.kernel()
            self.assertAlmostEqual(driver.ekin + driver.epot, beginning_energy,
                                   4)

    def test_ss_s0_ethylene_zero_init_veloc(self):
        driver = md.NVE(casscf_scanner, dt=5, steps=10)

        driver.kernel()

        self.assertAlmostEqual(driver.ekin, 0.0034505950738415096, 8)
        self.assertAlmostEqual(driver.epot, -78.05265768927349, 8)

        final_coord = np.array([[-0.0189651264, -0.0220674580, -0.0495315337],
                                [-0.0015076774, 0.0643680776, 2.5462148239],
                                [2.0038909173, 0.0058581097, -0.8642163260],
                                [1.8274638861, -0.2576472219, 3.6361858366],
                                [-1.7389508213, -0.2715870956, 3.7350500327],
                                [-1.8486454469, 0.0197089974, -1.0218233017]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))

if __name__ == "__main__":
    print("Full Tests for NVE Velocity Verlet")
    unittest.main()
