#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy

from pyscf.scf import cphf


class KnownValues(unittest.TestCase):
    def test_near_dependent_right_hand_sides(self):
        tol = 1e-9
        a = numpy.diag([.1, .2])
        mo_energy = numpy.array([0., 1., 1.])
        mo_occ = numpy.array([2., 0., 0.])
        mo1base = numpy.array([[1., 0.], [1., 3e-8]])

        def fvind(mo1):
            return numpy.asarray(mo1).reshape(-1, 2).dot(a.T)

        mo1, _ = cphf.solve(
            fvind, mo_energy, mo_occ, -mo1base.reshape(2, 2, 1), tol=tol
        )
        residual = abs(fvind(mo1) + mo1.reshape(-1, 2) - mo1base).max()
        self.assertLess(residual, tol)


if __name__ == "__main__":
    print("Full Tests for cphf")
    unittest.main()
