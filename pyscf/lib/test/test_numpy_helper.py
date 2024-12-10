# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
    def test_inplace_transpose_scale(self):
        a = numpy.random.random((5,5))
        acopy = a.copy()
        with lib.with_omp_threads(4):
            lib.transpose(a, inplace=True)
        self.assertAlmostEqual(abs(a.T - acopy).max(), 0, 12)

        a = numpy.random.random((405,405))-1j
        acopy = a.copy()
        lib.transpose(a, inplace=True)
        self.assertAlmostEqual(abs(a.T - acopy).max(), 0, 12)
        a[:] = acopy
        lib.inplace_transpose_scale(a, 1.5)
        self.assertAlmostEqual(abs(a.T - acopy*1.5).max(), 0, 12)
        a = numpy.random.random((2, 405, 405))
        acopy = a.copy()
        with lib.with_omp_threads(1):
            lib.transpose(a, axes=(0,2,1), inplace=True)
        self.assertAlmostEqual(abs(a - acopy.transpose(0,2,1)).max(), 0, 12)
        a[:] = acopy
        with lib.with_omp_threads(4):
            lib.transpose(a, axes=(0,2,1), inplace=True)
        self.assertAlmostEqual(abs(a - acopy.transpose(0,2,1)).max(), 0, 12)

    def test_transpose(self):
        a = numpy.random.random((400,900))
        self.assertAlmostEqual(abs(a.T - lib.transpose(a)).max(), 0, 12)
        b = a[:400,:400]
        c = numpy.copy(b)
        self.assertAlmostEqual(abs(b.T - lib.transpose(c,inplace=True)).max(), 0, 12)
        a = a.reshape(40,10,-1)
        self.assertAlmostEqual(abs(a.transpose(0,2,1) - lib.transpose(a,(0,2,1))).max(), 0, 12)

        a = (a*1000).astype(numpy.int32)
        at = lib.transpose(a)
        self.assertAlmostEqual(abs(a.T - at).max(), 0, 12)
        self.assertTrue(at.dtype == numpy.int32)

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

        a = (a*1000).astype(numpy.int32)
        b = a + a.T
        c = lib.transpose_sum(a)
        self.assertAlmostEqual(abs(b-c).max(), 0, 12)
        self.assertTrue(c.dtype == numpy.int32)

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
        self.assertAlmostEqual(lib.fp(lib.unpack_tril(a, axis=0)), -103.03970592075423, 10)

        a = numpy.zeros((5, 0))
        self.assertEqual(lib.unpack_tril(a, axis=-1).shape, (5, 0, 0))

        a = numpy.zeros((0, 5))
        self.assertEqual(lib.unpack_tril(a, axis=0).shape, (0, 0, 5))

    def test_unpack_tril_integer(self):
        a = lib.unpack_tril(numpy.arange(6, dtype=numpy.int32))
        self.assertTrue(a.dtype == numpy.int32)
        self.assertTrue(numpy.array_equal(a, numpy.array([(0,1,3),(1,2,4),(3,4,5)])))

    def test_pack_tril_integer(self):
        a = lib.pack_tril(numpy.arange(9, dtype=numpy.int32).reshape(3,3))
        self.assertTrue(numpy.array_equal(a, numpy.array((0,3,4,6,7,8))))
        self.assertTrue(a.dtype == numpy.int32)

    def test_unpack_row(self):
        row = numpy.arange(28.)
        ref = lib.unpack_tril(row)[4]
        self.assertTrue(numpy.array_equal(ref, lib.unpack_row(row, 4)))

        row = numpy.arange(28, dtype=numpy.int32)
        ref = lib.unpack_tril(row)[4]
        a = lib.unpack_row(row, 4)
        self.assertTrue(numpy.array_equal(ref, a))
        self.assertTrue(a.dtype == numpy.int32)

    def test_hermi_triu_int(self):
        a = numpy.arange(9).reshape(3,3)
        self.assertRaises(NotImplementedError, lib.hermi_triu, a)

    def test_take_2d(self):
        a = numpy.arange(49.).reshape(7,7)
        idx = [3,0,5]
        idy = [5,4,1]
        ref = a[idx][:,idy]
        self.assertTrue(numpy.array_equal(ref, lib.take_2d(a, idx, idy)))

        a = numpy.arange(49, dtype=numpy.int32).reshape(7,7)
        ref = a[idx][:,idy]
        self.assertTrue(numpy.array_equal(ref, lib.take_2d(a, idx, idy)))
        self.assertTrue(lib.take_2d(a, idx, idy).dtype == numpy.int32)

    def test_takebak_2d(self):
        b = numpy.arange(9.).reshape((3,3))
        a = numpy.arange(49.).reshape(7,7)
        idx = numpy.array([3,0,5])
        idy = numpy.array([5,4,1])
        ref = a.copy()
        ref[idx[:,None],idy] += b
        lib.takebak_2d(a, b, idx, idy)
        self.assertTrue(numpy.array_equal(ref, a))

        b = numpy.arange(9, dtype=numpy.int32).reshape((3,3))
        a = numpy.arange(49, dtype=numpy.int32).reshape(7,7)
        ref = a.copy()
        ref[idx[:,None],idy] += b
        lib.takebak_2d(a, b, idx, idy)
        self.assertTrue(numpy.array_equal(ref, a))

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

        def check(a, b):
            self.assertAlmostEqual(abs(a.dot(b) - lib.dot(a, b)).max(), 0, 13)
        dims = [4, 17, 70]
        for m in dims:
            for n in dims:
                for k in dims:
                    check(numpy.random.rand(m, k), numpy.random.rand(k, n))
                    check(numpy.random.rand(k, m).T, numpy.random.rand(k, n))
                    check(numpy.random.rand(m, k), numpy.random.rand(n, k).T)
                    check(numpy.random.rand(k, m).T, numpy.random.rand(n, k).T)

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

    def test_frompointer(self):
        s = numpy.ones(4, dtype=numpy.int16)
        ptr = s.ctypes.data
        a = lib.frompointer(ptr, count=2, dtype=numpy.int32)
        self.assertTrue(numpy.array_equal(a, [65537, 65537]))

    def test_split_reshape(self):
        numpy.random.seed(3)
        shapes = (numpy.random.random((5,4)) * 5).astype(int)
        ref = [numpy.random.random([x for x in shape if x > 1]) for shape in shapes]
        shapes = [x.shape for x in ref]
        a = numpy.hstack([x.ravel() for x in ref])
        a = lib.split_reshape(a, shapes)
        for x, y in zip(a, ref):
            self.assertAlmostEqual(abs(x-y).max(), 0, 12)

        b = lib.split_reshape(numpy.arange(17), ((2,3), (1,), ((2,2), (1,1))))

        self.assertRaises(ValueError, lib.split_reshape, numpy.arange(3), ((2,2),))
        self.assertRaises(ValueError, lib.split_reshape, numpy.arange(3), (2,2))

    def test_ndarray_pointer_2d(self):
        a = numpy.eye(3)
        addr = lib.ndarray_pointer_2d(a)
        self.assertTrue(all(addr == a.ctypes.data + numpy.array([0, 24, 48])))

    def test_omatcopy(self):
        a = numpy.random.random((5,5))
        b = numpy.empty_like(a)
        lib.omatcopy(a, out=b)
        self.assertTrue(numpy.all(a == b))
        a = numpy.random.random((403,410)).T
        b = numpy.empty_like(a)
        lib.omatcopy(a, out=b)
        self.assertTrue(numpy.all(a == b))

    def test_zeros(self):
        a = lib.zeros((100,100), dtype=numpy.double)
        self.assertTrue(numpy.all(a == 0))
        self.assertTrue(a.dtype == numpy.double)
        a = lib.zeros((100,100), dtype=numpy.complex128)
        self.assertTrue(numpy.all(a == 0))
        self.assertTrue(a.dtype == numpy.complex128)
        a = lib.zeros((100,100), dtype=numpy.int32)
        self.assertTrue(numpy.all(a == 0))
        self.assertTrue(a.dtype == numpy.int32)

    def test_entrywise_mul(self):
        a = numpy.random.random((101,100))
        b = numpy.random.random((101,100))
        prod = lib.entrywise_mul(a, b)
        self.assertTrue(numpy.allclose(prod, a * b))
        a = numpy.random.random((101,100))
        b = numpy.random.random((101,100))
        a = a + a*1j
        b = b + b*1j
        prod = lib.entrywise_mul(a, b)
        self.assertTrue(numpy.allclose(prod, a * b))
        # inplace test
        lib.entrywise_mul(a, b, out=b)
        self.assertTrue(numpy.allclose(prod, b))

if __name__ == "__main__":
    print("Full Tests for numpy_helper")
    unittest.main()
