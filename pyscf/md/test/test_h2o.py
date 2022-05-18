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
            [0.000613, 0.000080, 0.000792],
            [-0.000664, -0.000326, 0.002152],
            [-0.002068, -0.000332, 0.000086]])

        driver = integrator.VelocityVerlot(hf_scanner,
                                           mol=h2o,
                                           veloc=init_veloc,
                                           dt=10, max_iterations=10)

        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 0.02222510919429615, 12)
        self.assertAlmostEqual(driver.epot, -75.95967870293603, 12)

        final_coord = np.array([
            [ 0.0550940841,  0.0041888792,  0.0740675912],
            [-0.0580532687, -1.4128705258, 1.2618118745],
            [-0.1866218902,  1.4014392008, 1.1139054987]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        if CHECK_STABILITY:
            beginning_energy = driver.ekin + driver.epot

            driver.max_iterations=990
            driver.kernel()
            self.assertAlmostEqual(driver.ekin+driver.epot, beginning_energy, 4)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
