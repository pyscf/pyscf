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
from pyscf import lib

class KnownValues(unittest.TestCase):
    def test_call_in_background(self):
        def bg_raise():
            def raise1():
                raise ValueError

            with lib.call_in_background(raise1) as f:
                f()

            raise IndexError

        self.assertRaises(lib.ThreadRuntimeError, bg_raise)

    def test_index_tril_to_pair(self):
        i_j = (numpy.random.random((2,30)) * 100).astype(int)
        i0 = numpy.max(i_j, axis=0)
        j0 = numpy.min(i_j, axis=0)
        ij = i0 * (i0+1) // 2 + j0
        i1, j1 = lib.index_tril_to_pair(ij)
        self.assertTrue(numpy.all(i0 == i1))
        self.assertTrue(numpy.all(j0 == j1))

if __name__ == "__main__":
    unittest.main()
