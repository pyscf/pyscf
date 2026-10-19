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

    md.set_seed(0)

def tearDownModule():
    global o2
    del o2

class KnownValues(unittest.TestCase):

    def test_o2_mbveloc_high_cost(self):
        TEMPERATURE = 3000
        v = []

        for i in range(1000000):
            v.extend(md.distributions.MaxwellBoltzmannVelocity(o2, T=TEMPERATURE))

        v = np.array(v).flatten()
        v_squared = v**2
        self.assertAlmostEqual(np.mean(v), 0.0, 4)

        # <v^2> = kbT/m
        expected_v_squared = TEMPERATURE*data.nist.BOLTZMANN/data.nist.HARTREE2J/(data.elements.COMMON_ISOTOPE_MASSES[8]*data.nist.AMU2AU)
        self.assertAlmostEqual(np.mean(v_squared), expected_v_squared, 8)

    def test_temperature_and_mass_scaling(self):
        mol = gto.M(atom='H 0 0 0; O 0 0 1', spin=1, basis='sto-3g', verbose=0)
        cold = md.distributions.MaxwellBoltzmannVelocity(mol, 300, np.random.default_rng(4))
        hot = md.distributions.MaxwellBoltzmannVelocity(mol, 1200, np.random.default_rng(4))
        np.testing.assert_allclose(hot, 2*cold, atol=1e-15)
        np.testing.assert_array_equal(md.distributions.MaxwellBoltzmannVelocity(mol, 0), 0)
        rng = np.random.default_rng(8)
        samples = np.array([md.distributions.MaxwellBoltzmannVelocity(mol, 300, rng)
                            for _ in range(4000)])
        masses = np.array([data.elements.COMMON_ISOTOPE_MASSES[z] for z in mol.atom_charges()])
        variance = 300*data.nist.BOLTZMANN/data.nist.HARTREE2J/(masses*data.nist.AMU2AU)
        scaled = samples / np.sqrt(variance)[None, :, None]
        # Six standard errors, using 12000 independent Cartesian samples per atom.
        np.testing.assert_allclose(scaled.mean(axis=(0, 2)), 0, atol=6/np.sqrt(12000))
        np.testing.assert_allclose((scaled**2).mean(axis=(0, 2)), 1, atol=6*np.sqrt(2/12000))

if __name__ == "__main__":
    print("Full Tests for Distribution Sampling")
    unittest.main()
