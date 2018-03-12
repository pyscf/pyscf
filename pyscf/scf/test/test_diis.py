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
import numpy
from pyscf.scf import diis

class KnowValues(unittest.TestCase):
    def test_addis_minimize(self):
        numpy.random.seed(1)
        ds = numpy.random.random((4,2,2))
        fs = numpy.random.random((4,2,2))
        es = numpy.random.random(4)
        v, x = diis.adiis_minimize(ds, fs, -1)
        self.assertAlmostEqual(v, -0.44797757916272785, 9)

    def test_eddis_minimize(self):
        numpy.random.seed(1)
        ds = numpy.random.random((4,2,2))
        fs = numpy.random.random((4,2,2))
        es = numpy.random.random(4)
        v, x = diis.ediis_minimize(es, ds, fs)
        self.assertAlmostEqual(v, 0.31551563100606295, 9)


if __name__ == "__main__":
    print("Full Tests for DIIS")
    unittest.main()

