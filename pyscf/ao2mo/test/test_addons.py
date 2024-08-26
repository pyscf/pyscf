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
from pyscf import ao2mo

def setUpModule():
    global n, a1, a4, a8, a2ij, a2kl, b1, b4, b2ij, b2kl, c1, c2ij, d1, d2kl
    n = 10
    np = n*(n+1)//2
    a8 = numpy.random.random((np*(np+1)//2))
    a4 = numpy.empty((np,np))
    a1 = numpy.empty((n,n,n,n))

    idx, idy = numpy.tril_indices(n)
    idxy = numpy.empty((n,n), dtype=int)
    idxy[idx,idy] = idxy[idy,idx] = numpy.arange(n*(n+1)//2)

    xx, yy = numpy.tril_indices(np)
    a4[xx,yy] = a4[yy,xx] = a8

    idx, idy = numpy.tril_indices(n)
    idxy = numpy.empty((n,n), dtype=int)
    idxy[idx,idy] = idxy[idy,idx] = numpy.arange(n*(n+1)//2)
    a2ij = a4[:,idxy]
    a2kl = a4[idxy]
    a1 = a2ij[idxy]

    b4 = numpy.random.random((np,np))
    b1 = b4[:,idxy][idxy]
    b2ij = b4[:,idxy]
    b2kl = b4[idxy]
    b1 = b2ij[idxy]

    c2ij = numpy.random.random((np,n,n))
    c1 = c2ij[idxy]

    d2kl = numpy.random.random((n,n,np))
    d1 = d2kl[:,:,idxy]

def tearDownModule():
    global n, a1, a4, a8, a2ij, a2kl, b1, b4, b2ij, b2kl, c1, c2ij, d1, d2kl
    del n, a1, a4, a8, a2ij, a2kl, b1, b4, b2ij, b2kl, c1, c2ij, d1, d2kl

class KnownValues(unittest.TestCase):
    def test_restore8(self):
        self.assertTrue(numpy.allclose(a8, ao2mo.restore(8, a1, n)))
        self.assertTrue(numpy.allclose(a8, ao2mo.restore('8', a4, n)))
        self.assertTrue(numpy.allclose(a8, ao2mo.restore('s8', a8, n)))
        self.assertTrue(numpy.allclose(a8, ao2mo.restore('s8', a2ij, n)))
        self.assertTrue(numpy.allclose(a8, ao2mo.restore('s8', a2kl, n)))

    def test_restore4(self):
        self.assertTrue(numpy.allclose(a4, ao2mo.restore(4, a1, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore('4', a4, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore('s4', a8, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore('s4', a2ij, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore('s4', a2kl, n)))

        self.assertTrue(numpy.allclose(b4, ao2mo.restore(4, b1, n)))
        self.assertTrue(numpy.allclose(b4, ao2mo.restore('4', b4, n)))
        self.assertTrue(numpy.allclose(b4, ao2mo.restore('4', b2ij, n)))
        self.assertTrue(numpy.allclose(b4, ao2mo.restore('4', b2kl, n)))

    def test_restore1(self):
        self.assertTrue(numpy.allclose(a1, ao2mo.restore(1, a1, n)))
        self.assertTrue(numpy.allclose(a1, ao2mo.restore('1', a4, n)))
        self.assertTrue(numpy.allclose(a1, ao2mo.restore('s1', a8, n)))
        self.assertTrue(numpy.allclose(a1, ao2mo.restore('s1', a2ij, n)))
        self.assertTrue(numpy.allclose(a1, ao2mo.restore('s1', a2kl, n)))

        self.assertTrue(numpy.allclose(b1, ao2mo.restore(1, b1, n)))
        self.assertTrue(numpy.allclose(b1, ao2mo.restore('1', b4, n)))
        self.assertTrue(numpy.allclose(b1, ao2mo.restore('1', b2ij, n)))
        self.assertTrue(numpy.allclose(b1, ao2mo.restore('1', b2kl, n)))

        self.assertTrue(numpy.allclose(c1, ao2mo.restore(1, c1, n)))
        self.assertTrue(numpy.allclose(c1, ao2mo.restore('1', c2ij, n)))

        self.assertTrue(numpy.allclose(d1, ao2mo.restore('1', d1, n)))
        self.assertTrue(numpy.allclose(d1, ao2mo.restore('1', d2kl, n)))

    def test_restore_s2ij(self):
        self.assertTrue(numpy.allclose(a2ij, ao2mo.restore('s2ij', a1, n)))
        self.assertTrue(numpy.allclose(a2ij, ao2mo.restore('s2ij', a4, n)))
        self.assertTrue(numpy.allclose(a2ij, ao2mo.restore('s2ij', a8, n)))
        self.assertTrue(numpy.allclose(a2ij, ao2mo.restore('s2ij', a2ij, n)))
        self.assertTrue(numpy.allclose(a2ij, ao2mo.restore('s2ij', a2kl, n)))

        self.assertTrue(numpy.allclose(b2ij, ao2mo.restore('s2ij', b1, n)))
        self.assertTrue(numpy.allclose(b2ij, ao2mo.restore('s2ij', b4, n)))
        self.assertTrue(numpy.allclose(b2ij, ao2mo.restore('s2ij', b2ij, n)))
        self.assertTrue(numpy.allclose(b2ij, ao2mo.restore('s2ij', b2kl, n)))

        self.assertTrue(numpy.allclose(c2ij, ao2mo.restore('s2ij', c1, n)))
        self.assertTrue(numpy.allclose(c2ij, ao2mo.restore('s2ij', c2ij, n)))

    def test_restore_s2kl(self):
        self.assertTrue(numpy.allclose(a2kl, ao2mo.restore('s2kl', a1, n)))
        self.assertTrue(numpy.allclose(a2kl, ao2mo.restore('s2kl', a4, n)))
        self.assertTrue(numpy.allclose(a2kl, ao2mo.restore('s2kl', a8, n)))
        self.assertTrue(numpy.allclose(a2kl, ao2mo.restore('s2kl', a2ij, n)))
        self.assertTrue(numpy.allclose(a2kl, ao2mo.restore('s2kl', a2kl, n)))

        self.assertTrue(numpy.allclose(b2kl, ao2mo.restore('s2kl', b1, n)))
        self.assertTrue(numpy.allclose(b2kl, ao2mo.restore('s2kl', b4, n)))
        self.assertTrue(numpy.allclose(b2kl, ao2mo.restore('s2kl', b2ij, n)))
        self.assertTrue(numpy.allclose(b2kl, ao2mo.restore('s2kl', b2kl, n)))

        self.assertTrue(numpy.allclose(d2kl, ao2mo.restore('s2kl', d1, n)))
        self.assertTrue(numpy.allclose(d2kl, ao2mo.restore('s2kl', d2kl, n)))

    def test_load(self):
        a = numpy.zeros(3)
        with ao2mo.load(a) as eri:
            self.assertTrue(a is eri)


if __name__ == '__main__':
    print('Full Tests for ao2mo.addons')
    unittest.main()
