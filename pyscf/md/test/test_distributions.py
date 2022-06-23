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

from pyscf import gto, data
import pyscf.md as md

def setUpModule():
    global o2
    o2 = gto.M(verbose=3,
                atom=[['O', 0, 0, 0], ['O', 0, 0, 0.587]],
                basis='def2-svp')



def tearDownModule():
    global o2
    del o2

class KnownValues(unittest.TestCase):

    def test_water_MB_velocity(self):
        TEMPERATURE = 100000
        v = []

        for i in range(10000):
            v.extend(md.distributions.MaxwellBoltzmannVelocity(o2, T=TEMPERATURE))

        v = np.array(v).flatten()
        v_squared = v**2
        self.assertAlmostEqual(np.mean(v), 0.0, 2)

        # <v^2> = kbT/m
        expected_v_squared = TEMPERATURE*data.nist.BOLTZMANN/data.nist.HARTREE2J/data.elements.COMMON_ISOTOPE_MASSES[8]
        self.assertAlmostEqual(np.mean(v_squared), expected_v_squared, 3)

if __name__ == "__main__":
    print("Full Tests for Distribution Sampling")
    unittest.main()