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

ethylene = gto.M(
    verbose=3,
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
ncas, nelecas = (2,2)
casscf_scanner = hf.CASSCF(ncas, nelecas)
casscf_scanner.conv_tol = 1e-12
casscf_scanner.conv_tol_grad = 1e-6

def tearDownModule():
    global ethylene, hf, casscf_scanner
    ethylene.stdout.close()
    del ethylene, hf, casscf_scanner

class KnownValues(unittest.TestCase):
    def test_ss_s0_zero_init_veloc(self):
        driver = integrator.VelocityVerlot(casscf_scanner, dt=5, max_iterations=100)
        driver.energy_output = "BOMD.md.energies"

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
    print("Full Tests for ethylene")
    unittest.main()
