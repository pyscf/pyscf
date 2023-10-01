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
from pyscf import gto, scf, data
import pyscf.md as md

def setUpModule():
    global h2o, h2o_scanner, o2, o2_scanner
    h2o = gto.M(verbose=3,
                output='/dev/null',
                atom='''O -2.9103342153    -15.4805607073    -14.9344021104
                        H -2.5833611256    -14.8540450112    -15.5615823519
                       H  -2.7404195919    -16.3470417109    -15.2830799053''',
                basis='def2-svp')

    h2o_scanner = scf.RHF(h2o)
    h2o_scanner.build()
    h2o_scanner.conv_tol_grad = 1e-6
    h2o_scanner.max_cycle = 700

    o2 = gto.M(verbose=3,
               output='/dev/null',
               atom='''O 0	0	0.0298162549
               O 0	0	1.1701837504''',
               basis='def2-svp')

    o2_scanner = scf.RHF(o2)
    o2_scanner.build()
    o2_scanner.conv_tol_grad = 1e-6
    o2_scanner.max_cycle = 700


def tearDownModule():
    global h2o, h2o_scanner, o2, o2_scanner
    h2o_scanner.stdout.close()
    o2_scanner.stdout.close()
    del h2o, h2o_scanner, o2, o2_scanner


class KnownValues(unittest.TestCase):

    def test_temperature_non_linear(self):
        # Property Checked: Non-Linear Molecule Temperature
        # unit-converted velocities temp = 298.13 obtained from ORCA
        init_veloc = np.array([[-3.00670625e-05, -2.47219610e-04, -2.99235779e-04],
                               [-5.66022419e-05, -9.83256521e-04, -8.35299245e-04],
                               [-4.38260913e-04, -3.17970694e-04,  5.07818817e-04]])

        driver = md.integrators.NVTBerendson(h2o_scanner, veloc=init_veloc,
                                             dt=20, steps=50, T=300, taut=413)

        driver._masses = np.array(
                [data.elements.COMMON_ISOTOPE_MASSES[m] * data.nist.AMU2AU
                 for m in driver.mol.atom_charges()])
        driver.ekin = driver.compute_kinetic_energy()
        self.assertAlmostEqual(driver.temperature(), 298.13, delta=0.2)

    def test_temperature_linear(self):
        # Property Checked: Linear Molecule Temperature
        # unit-converted velocities temp = 468.31 obtained from ORCA
        init_veloc = np.array([[-0.,          0.,          0.00039058],
                               [ 0.,         -0.,         -0.00039058]])

        driver = md.integrators.NVTBerendson(o2_scanner, veloc=init_veloc,
                                             dt=20, steps=50, T=300, taut=413)

        driver._masses = np.array(
                [data.elements.COMMON_ISOTOPE_MASSES[m] * data.nist.AMU2AU
                 for m in driver.mol.atom_charges()])
        driver.ekin = driver.compute_kinetic_energy()
        self.assertAlmostEqual(driver.temperature(), 468.31, delta=0.2)

if __name__ == "__main__":
    print("Property Tests")
    unittest.main()
