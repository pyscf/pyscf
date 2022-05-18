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
        driver = integrator.VelocityVerlot(casscf_scanner, dt=5, max_iterations=10)
        return
        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 0.0034505950754127246, 12)
        self.assertAlmostEqual(driver.epot, -78.05265768927464, 12)

        final_coord = np.array([
            [-0.0189651263, -0.0220674578, -0.0495315336],
            [-0.0015076774,  0.0643680776,  2.5462148239],
            [ 2.0038909173,  0.0058581090, -0.8642163262],
            [ 1.8274638862, -0.2576472221,  3.6361858368],
            [-1.7389508212, -0.2715870959,  3.7350500325],
            [-1.8486454478,  0.0197089966, -1.0218233020]])


        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))


    def test_sa_s1_init_veloc(self):
        init_veloc = np.array([
            [ 0.00000952, -0.00028562, -0.00004197],
            [-0.00004845,  0.00022793,  0.00023270],
            [-0.00054174,  0.00086082, -0.00110483],
            [-0.00109802, -0.00061562,  0.00028189],
            [ 0.00166588, -0.00063874, -0.00173299],
            [ 0.00043740,  0.00108051,  0.00028496]])

        n_states = 3
        sa_scanner = casscf_scanner.set(natorb=True).state_average_([1.0/float(n_states),]*n_states)
        sa_scanner.spin = 0
        sa_scanner.fix_spin_(ss=0)

        sa_scanner.conv_tol = sa_scanner.conv_tol_diabatize = 1e-12
        sa_scanner.conv_tol_grad = 1e-6
        sa_scanner = sa_scanner.nuc_grad_method().as_scanner(state=1)
        driver = integrator.VelocityVerlot(sa_scanner, dt=5, max_iterations=100, veloc=init_veloc)
        
        driver.energy_output='BOMD.md.energies'
        driver.trajectory_output='BOMD.md.xyz'
        driver.kernel()

if __name__ == "__main__":
    print("Full Tests for ethylene")
    unittest.main()
