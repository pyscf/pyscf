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
# Author: Aniruddha Seal <aniruddhaseal2011@gmail.com>

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

def tearDownModule():
    global h2o, hf_scanner
    hf_scanner.stdout.close()
    del h2o, hf_scanner

class KnownValues(unittest.TestCase):

    def test_hf_water_init_veloc(self):
        init_veloc = np.array([[0.000336, 0.000044, 0.000434],
                               [-0.000364, -0.000179, 0.001179],
                               [-0.001133, -0.000182, 0.000047]])

        driver = md.integrators.NVTBerendson(hf_scanner, veloc=init_veloc,
        				       dt=5, steps=20, T=165, taut=50)

        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 0.007071731316944, 8)
        self.assertAlmostEqual(driver.epot, -75.96084293823, 8)

        final_coord = np.array([[0.03203339, 0.00219695, 0.04414054],
 				 [-0.03398398, -1.40833795, 1.18785634],
 				 [-0.10817007, 1.40567903, 1.10489554]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        if CHECK_STABILITY:

            driver.steps = 990
            driver.kernel()
            self.assertTrue((driver.T - 5)<= driver.temperature() <= (driver.T + 5))

if __name__ == "__main__":
    print("Full Tests for NVT Berendson Thermostat")
    unittest.main()
