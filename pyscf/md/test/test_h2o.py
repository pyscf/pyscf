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
import os
import numpy as np
from pyscf import gto, scf
import pyscf.md.integrator as integrator


h2o = gto.M(
    verbose=5,
    output='/dev/null',
    atom=[
    ['O', 0,0,0],
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
        self.assertAlmostEqual(driver.ekin, 0.0003490353178906977,8)
        self.assertAlmostEqual(driver.epot, -75.96132726509461, 8)

        final_coord = np.array([[-7.28115148e-19,  1.49705405e-16,  2.05638219e-03],
                       [-9.60088675e-17, -1.41129779e+00,  1.09281818e+00],
                       [ 1.07658710e-16,  1.41129779e+00,  1.09281818e+00]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        beginning_energy = driver.ekin + driver.epot
    #     #driver.max_iterations=990
    #     #driver.kernel()
    #     #self.assertAlmostEqual(driver.ekin+driver.epot, beginning_energy, 6)

    def test_init_veloc(self):
        init_veloc = np.array([
            [0.000335,   0.000044,   0.000432],
            [-0.000363,  -0.000178,   0.001175],
            [-0.001129,  -0.000181,   0.000047]
        ])

        driver = integrator.VelocityVerlot(hf_scanner,
                                           mol=h2o,
                                           veloc = init_veloc,
                                           dt=10, max_iterations=10,
                                           energy_output="BOMD.md.energies",
                                           trajectory_output="BOMD.md.xyz")

        driver.kernel()
        print(driver.ekin)
        driver.energy_output.close()
        driver.trajectory_output.close()

if __name__=="__main__":
    print("Full Tests for H2O")
    unittest.main()
