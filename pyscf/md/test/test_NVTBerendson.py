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
    global h2o, hf_scanner
    h2o = gto.M(verbose=3,
                output='/dev/null',
                atom='''O -2.9103342153    -15.4805607073    -14.9344021104
                        H -2.5833611256    -14.8540450112    -15.5615823519
                       H  -2.7404195919    -16.3470417109    -15.2830799053''',
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
        init_veloc = np.array([[-3.00670625e-05, -2.47219610e-04, -2.99235779e-04],
                               [-5.66022419e-05, -9.83256521e-04, -8.35299245e-04],
                               [-4.38260913e-04, -3.17970694e-04,  5.07818817e-04]])

        driver = md.integrators.NVTBerendson(hf_scanner, veloc=init_veloc,
                                             dt=20, steps=20, T=300, taut=413)

        driver.kernel()
        self.assertAlmostEqual(driver.ekin, 4.244286252900E-03, 8)
        self.assertAlmostEqual(driver.epot, -7.596117676337E+01, 8)

        final_coord = np.array([[ -5.51486188, -29.3425402,  -28.32832762],
                                [ -4.8843336,  -28.48585797, -29.75984939],
                                [ -5.30517791, -31.05471672, -28.76717135]])

        self.assertTrue(np.allclose(driver.mol.atom_coords(), final_coord))

        if CHECK_STABILITY:

            driver.steps = 990
            driver.kernel()
            self.assertAlmostEqual(driver.temperature(), driver.T, delta=5)

if __name__ == "__main__":
    print("Full Tests for NVT Berendson Thermostat")
    unittest.main()
