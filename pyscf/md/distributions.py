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

import numpy as np

from pyscf import data, md

def MaxwellBoltzmannVelocity(mol, T=298.15, rng=md.rng):
    """Computes velocities for a molecular structure using
        a Maxwell-Boltzmann distribution.
        Args:
            mol : gto.mol object

            T : float
                Temperature, in Kelvin, that the distribution represents.

            rng : np.random.Generator
                Random number generator to sample from. Must contain a method
                `normal`. Default is to use the md.rng which is a
                np.random.Generator

        Returns:
            Velocities as a ndarray of dimension (natm, 3) in atomic units.
        """

    veloc = []
    Tkb = T*data.nist.BOLTZMANN/data.nist.HARTREE2J
    MEAN = 0.0

    for m in mol.atom_charges():
        m = data.elements.COMMON_ISOTOPE_MASSES[m] * data.nist.AMU2AU
        sigma = np.sqrt(Tkb/m)

        veloc.append(rng.normal(loc=MEAN, scale=sigma, size=3))

    return np.array(veloc)
