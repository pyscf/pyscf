#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import gto, scf
import pyscf.md.integrator as integrator

CHECK_STABILITY = False

h2o = gto.M(
    verbose=3,
    #output='/dev/null',
    atom=[
        ['O', 0, 0, 0],
        ['H', 0, -0.757, 0.587],
        ['H', 0, 0.757, 0.587]
    ],
    basis='def2-svp')

hf_scanner = scf.RHF(h2o)
hf_scanner.build()
hf_scanner.conv_tol_grad = 1e-6
hf_scanner.max_cycle = 700

def tearDownModule():
    global h2o, hf_scanner
    h2o.stdout.close()
    hf_scanner.stdout.close()
    del h2o, hf_scanner

class KnownValues(unittest.TestCase):
    def test_zero_init_veloc(self):
        driver = integrator.VelocityVerlot(hf_scanner, dt=10, max_iterations=10)
        
        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 0.000349066856492198, 12)
        self.assertAlmostEqual(driver.epot, -75.96132729628864, 12)

        final_coord = np.array([
            [0.0000000000,  0.0000000000, 0.0020720320],
            [0.0000000000, -1.4113069887, 1.0928269088],
            [0.0000000000,  1.4113069887, 1.0928269088]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        if CHECK_STABILITY:
            beginning_energy = driver.ekin + driver.epot
            driver.max_iterations=990
            driver.kernel()
            self.assertAlmostEqual(driver.ekin+driver.epot, beginning_energy, 4)

    def test_init_veloc(self):
        init_veloc = np.array([
            [ 0.000336,   0.000044,   0.000434],
            [-0.000364,  -0.000179,   0.001179],
            [-0.001133,  -0.000182,   0.000047]])


        driver = integrator.VelocityVerlot(hf_scanner,
                                           mol=h2o,
                                           veloc=init_veloc,
                                           dt=5, max_iterations=10)

        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 0.0068732364518669445, 12)
        self.assertAlmostEqual(driver.epot, -75.96078835576954, 12)

        final_coord = np.array([
            [ 0.0151120306,  0.0017437807,  0.0201833153],
            [-0.0163089669, -1.4306397410,  1.1556586744],
            [-0.0509295530,  1.4181437135,  1.1076812077]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        if CHECK_STABILITY:
            beginning_energy = driver.ekin + driver.epot

            driver.max_iterations=990
            driver.kernel()
            self.assertAlmostEqual(driver.ekin+driver.epot, beginning_energy, 4)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
