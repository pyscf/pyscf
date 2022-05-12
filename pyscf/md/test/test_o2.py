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

o2 = gto.M(
    verbose=3,
#    output='o2.log',
    atom='O 0 0 0; O 0 0 1.2',
    basis='ccpvdz',
    spin=2)

hf = o2.RHF().run()
ncas, nelecas = (6,8)
casscf_scanner = hf.CASSCF(ncas, nelecas)
casscf_scanner.conv_tol = 1e-12

def tearDownModule():
    global o2, hf, casscf_scanner
    o2.stdout.close()
    del o2, hf, casscf_scanner

class KnownValues(unittest.TestCase):
    def test_zero_init_veloc(self):
        driver = integrator.VelocityVerlot(casscf_scanner, dt=10, max_iterations=10)
        driver.trajectory_output = "BOMD.md.xyz"

        driver.kernel()
        #self.assertAlmostEqual(driver.ekin, 0.0003490772304325728, 12)
        #self.assertAlmostEqual(driver.epot, -75.96132730618872, 12)

        #final_coord = np.array([
        #    [-0.0000000000, 0.0000000000, 0.0020715828],
        #    [-0.0000000000, -1.4113094571, 1.0928291295],
        #    [0.0000000000, 1.4113094571, 1.0928291295]])

        #self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))
        #if CHECK_STABILITY:
        #    beginning_energy = driver.ekin + driver.epot
        #    driver.max_iterations=990
        #    driver.kernel()
        #    self.assertAlmostEqual(driver.ekin+driver.epot, beginning_energy, 4)

if __name__ == "__main__":
    print("Full Tests for O2")
    unittest.main()
