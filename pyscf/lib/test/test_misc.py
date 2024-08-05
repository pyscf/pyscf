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
    def test_call_in_background_skip(self):
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

    def test_class_as_method(self):
        class A:
            def f1(self):
                return 'a'
            f2 = lib.alias(f1)
        class B(A):
            def f1(self):
                return 'b'
        b = B()
        self.assertEqual(b.f2(), 'b')

    def test_isinteger(self):
        isinteger = lib.isinteger
        self.assertTrue(isinteger(0))
        self.assertTrue(isinteger(20))
        self.assertTrue(isinteger(-10))
        self.assertTrue(isinteger(numpy.int_(1.0)))
        self.assertFalse(isinteger(1.0))
        self.assertFalse(isinteger('1'))
        self.assertFalse(isinteger(True))

    def test_issequence(self):
        issequence = lib.issequence
        self.assertTrue(issequence([1, 2, 3]))
        self.assertTrue(issequence(numpy.array([1, 2, 3])))
        self.assertTrue(issequence(range(5)))
        self.assertTrue(issequence('abcde'))
        self.assertTrue(issequence(()))
        self.assertFalse(issequence(True))
        self.assertFalse(issequence(2.0))
        self.assertFalse(issequence(1))
        self.assertFalse(issequence({}))
        self.assertFalse(issequence(set()))

    def test_isintsequence(self):
        isintsequence = lib.isintsequence
        self.assertTrue(isintsequence([2, 4, 6]))
        self.assertTrue(isintsequence(numpy.array([2, 4, 6])))
        self.assertTrue(isintsequence([]))
        self.assertFalse(isintsequence([2.0, 4.0, 6.0]))
        self.assertFalse(isintsequence(numpy.array([2.0, 4.0, 6.0])))
        self.assertFalse(isintsequence((True, False)))
        self.assertFalse(isintsequence('123'))
        self.assertFalse(isintsequence(5))

    def test_prange_split(self):
        self.assertEqual(list(lib.prange_split(10, 3)), [(0, 4), (4, 7), (7, 10)])

    def test_pickle(self):
        import pickle
        from pyscf import gto
        mol = gto.M()
        mf = mol.GKS(xc='pbe')
        pickle.loads(pickle.dumps(mf))


if __name__ == "__main__":
    unittest.main()
