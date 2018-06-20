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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import itertools
from pyscf import lib

class KnownValues(unittest.TestCase):
    def test_transpose(self):
        a = numpy.random.random((400,900))
        self.assertAlmostEqual(abs(a.T - lib.transpose(a)).max(), 0, 12)
        b = a[:400,:400]
        c = numpy.copy(b)
        self.assertAlmostEqual(abs(b.T - lib.transpose(c,inplace=True)).max(), 0, 12)
        a = a.reshape(40,10,-1)
        self.assertAlmostEqual(abs(a.transpose(0,2,1) - lib.transpose(a,(0,2,1))).max(), 0, 12)

    def test_transpose_sum(self):
        a = numpy.random.random((3,400,400))
        self.assertAlmostEqual(abs(a[0]+a[0].T - lib.hermi_sum(a[0])).max(), 0, 12)
        self.assertAlmostEqual(abs(a+a.transpose(0,2,1) - lib.hermi_sum(a,(0,2,1))).max(), 0, 12)
        self.assertAlmostEqual(abs(a+a.transpose(0,2,1) - lib.hermi_sum(a,(0,2,1), inplace=True)).max(), 0, 12)
        a = numpy.random.random((3,400,400)) + numpy.random.random((3,400,400)) * 1j
        self.assertAlmostEqual(abs(a[0]+a[0].T.conj() - lib.hermi_sum(a[0])).max(), 0, 12)
        self.assertAlmostEqual(abs(a+a.transpose(0,2,1).conj() - lib.hermi_sum(a,(0,2,1))).max(), 0, 12)
        self.assertAlmostEqual(abs(a+a.transpose(0,2,1) - lib.hermi_sum(a,(0,2,1),hermi=3)).max(), 0, 12)
        self.assertAlmostEqual(abs(a+a.transpose(0,2,1).conj() - lib.hermi_sum(a,(0,2,1),inplace=True)).max(), 0, 12)

        a = numpy.random.random((400,400))
        b = a + a.T.conj()
        c = lib.transpose_sum(a)
        self.assertAlmostEqual(abs(b-c).max(), 0, 12)

    def test_unpack(self):
        a = numpy.random.random((400,400))
        a = a+a*.5j
        for i in range(400):
            a[i,i] = a[i,i].real
        b = a-a.T.conj()
        b = numpy.array((b,b))
        x = lib.hermi_triu(b[0].T, hermi=2, inplace=0)
        self.assertAlmostEqual(abs(b[0].T-x).max(), 0, 12)

        x = lib.hermi_triu(b[1], hermi=2, inplace=0)
        self.assertAlmostEqual(abs(b[1]-x).max(), 0, 12)
        self.assertAlmostEqual(abs(x - lib.unpack_tril(lib.pack_tril(x), 2)).max(), 0, 12)

        x = lib.hermi_triu(a, hermi=1, inplace=0)
        self.assertAlmostEqual(abs(x-x.T.conj()).max(), 0, 12)

        xs = numpy.asarray((x,x,x))
        self.assertAlmostEqual(abs(xs - lib.unpack_tril(lib.pack_tril(xs))).max(), 0, 12)

        numpy.random.seed(1)
        a = numpy.random.random((5050,20))
        self.assertAlmostEqual(lib.finger(lib.unpack_tril(a, axis=0)), -103.03970592075423, 12)

    def test_dot(self):
        a = numpy.random.random((400,400))
        b = numpy.random.random((400,400))
        self.assertAlmostEqual(abs(lib.dot(a  ,b  )-numpy.dot(a  ,b  )).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(a  ,b.T)-numpy.dot(a  ,b.T)).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(a.T,b  )-numpy.dot(a.T,b  )).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(a.T,b.T)-numpy.dot(a.T,b.T)).max(), 0, 12)

        a = numpy.random.random((400,40))
        b = numpy.random.random((40,400))
        self.assertAlmostEqual(abs(lib.dot(a  ,b  )-numpy.dot(a  ,b  )).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(b  ,a  )-numpy.dot(b  ,a  )).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(a.T,b.T)-numpy.dot(a.T,b.T)).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(b.T,a.T)-numpy.dot(b.T,a.T)).max(), 0, 12)
        a = numpy.random.random((400,40))
        b = numpy.random.random((400,40))
        self.assertAlmostEqual(abs(lib.dot(a  ,b.T)-numpy.dot(a  ,b.T)).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(b  ,a.T)-numpy.dot(b  ,a.T)).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(a.T,b  )-numpy.dot(a.T,b  )).max(), 0, 12)
        self.assertAlmostEqual(abs(lib.dot(b.T,a  )-numpy.dot(b.T,a  )).max(), 0, 12)

        a = numpy.random.random((400,400))
        b = numpy.random.random((400,400))
        c = numpy.random.random((400,400))
        d = numpy.random.random((400,400))
        self.assertTrue(numpy.allclose(numpy.dot(a+b*1j, c+d*1j), lib.dot(a+b*1j, c+d*1j)))
        self.assertTrue(numpy.allclose(numpy.dot(a, c+d*1j), lib.dot(a, c+d*1j)))
        self.assertTrue(numpy.allclose(numpy.dot(a+b*1j, c), lib.dot(a+b*1j, c)))

    def test_cartesian_prod(self):
        arrs = (range(3,9), range(4))
        cp = lib.cartesian_prod(arrs)
        for i,x in enumerate(itertools.product(*arrs)):
            self.assertTrue(numpy.allclose(x,cp[i]))

    def test_condense(self):
        locs = numpy.arange(5)
        a = numpy.random.random((locs[-1],locs[-1])) - .5
        self.assertTrue(numpy.allclose(a, lib.condense('sum', a, locs)))
        self.assertTrue(numpy.allclose(a, lib.condense('max', a, locs)))
        self.assertTrue(numpy.allclose(a, lib.condense('min', a, locs)))
        self.assertTrue(numpy.allclose(abs(a), lib.condense('abssum', a, locs)))
        self.assertTrue(numpy.allclose(abs(a), lib.condense('absmax', a, locs)))
        self.assertTrue(numpy.allclose(abs(a), lib.condense('absmin', a, locs)))
        self.assertTrue(numpy.allclose(abs(a), lib.condense('norm', a, locs)))

    def test_expm(self):
        a = numpy.random.random((300,300)) * .1
        a = a - a.T
        self.assertAlmostEqual(abs(scipy.linalg.expm(a) - lib.expm(a)).max(), 0, 12)

    def test_einsum(self):
        a = numpy.random.random((30,40,5,10))
        b = numpy.random.random((10,30,5,20))
        c = numpy.random.random((10,20,20))
        d = numpy.random.random((20,10))
        f = lib.einsum('ijkl,xiky,ayp,px->ajl', a,b,c,d, optimize=True)
        ref = lib.einsum('ijkl,xiky->jlxy', a,b)
        ref = lib.einsum('jlxy,ayp->jlxap', ref,c)
        ref = lib.einsum('jlxap,px->ajl', ref,d)
        self.assertAlmostEqual(abs(ref-f).max(), 0, 9)

        f = lib.einsum('ijkl,xiky,lyp,px->jl', a,b,c,d, optimize=True)
        ref = lib.einsum('ijkl,xiky->jlxy', a, b)
        ref = lib.einsum('jlxy,lyp->jlxp', ref, c)
        ref = lib.einsum('jlxp,px->jl', ref, d)
        self.assertAlmostEqual(abs(ref-f).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for numpy_helper")
    unittest.main()
