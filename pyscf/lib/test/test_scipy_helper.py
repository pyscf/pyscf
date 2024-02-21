# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy
from pyscf.lib import scipy_helper


class KnownValues(unittest.TestCase):

    def setUp(self):
        self.pivoted_cholesky = [scipy_helper.pivoted_cholesky,
                                 scipy_helper.pivoted_cholesky_python]

    def test_pivoted_cholesky_1x1(self):
        for func in self.pivoted_cholesky:
            A = numpy.array([[9.0]])
            L, piv, rank = func(A)
            self.assertEqual(L.shape, (1, 1))
            self.assertEqual(piv.shape, (1,))
            self.assertAlmostEqual(L[0, 0], 3.0, delta=1.0e-14)
            self.assertEqual(piv[0], 0)
            self.assertEqual(rank, 1)

    def test_pivoted_cholesky_2x2(self):
        for func in self.pivoted_cholesky:
            A = numpy.array([[9.0, 6.0], [6.0, 5.0]])
            L, piv, rank = func(A)
            L_ref = numpy.array([[3.0, 2.0], [0.0, 1.0]])
            piv_ref = numpy.array([0, 1])
            self.assertTrue(numpy.allclose(L, L_ref, atol=1.0e-14))
            self.assertTrue(numpy.array_equal(piv, piv_ref))
            self.assertEqual(rank, 2)

    def test_pivoted_cholesky_2x2_singular(self):
        for func in self.pivoted_cholesky:
            A = numpy.array([[1.0, 2.0], [2.0, 4.0]])
            L, piv, rank = func(A)
            L_ref = numpy.array([[2.0, 1.0], [0.0, 0.0]])
            piv_ref = numpy.array([1, 0])
            self.assertTrue(numpy.allclose(L, L_ref, atol=1.0e-14))
            self.assertTrue(numpy.array_equal(piv, piv_ref))
            self.assertEqual(rank, 1)

    def test_pivoted_cholesky_10x10(self):
        for func in self.pivoted_cholesky:
            # Positive-definite 10x10 matrix A
            A = numpy.array([[4.78, 3.56, 2.95, 3.16, 3.38, 3.10, 3.47, 2.77, 3.68, 4.29],
                            [3.56, 3.78, 2.50, 2.57, 2.44, 2.61, 3.21, 2.03, 3.19, 3.41],
                            [2.95, 2.50, 3.44, 2.44, 2.55, 2.30, 3.37, 2.09, 2.78, 3.15],
                            [3.16, 2.57, 2.44, 3.03, 2.21, 1.96, 2.67, 1.92, 2.45, 3.02],
                            [3.38, 2.44, 2.55, 2.21, 3.31, 2.13, 2.61, 2.31, 2.68, 2.99],
                            [3.10, 2.61, 2.30, 1.96, 2.13, 2.87, 2.72, 2.19, 2.87, 3.4 ],
                            [3.47, 3.21, 3.37, 2.67, 2.61, 2.72, 4.33, 2.38, 3.58, 3.85],
                            [2.77, 2.03, 2.09, 1.92, 2.31, 2.19, 2.38, 2.35, 2.39, 3.07],
                            [3.68, 3.19, 2.78, 2.45, 2.68, 2.87, 3.58, 2.39, 3.59, 3.79],
                            [4.29, 3.41, 3.15, 3.02, 2.99, 3.40, 3.85, 3.07, 3.79, 4.67]])
            L, piv, rank = func(A, lower=True)
            self.assertEqual(rank, 10)
            # Check if L^T * L == P^T * A * P
            P = numpy.zeros((10, 10))
            P[piv, numpy.arange(10)] = 1
            PtAP = numpy.linalg.multi_dot([P.T, A, P])
            LtL = numpy.dot(L, L.T)
            for i in range(10):
                for j in range(i+1, 10):
                    self.assertEqual(L[i, j], 0)
            self.assertTrue(numpy.allclose(LtL, PtAP, atol=1.0e-12))

    def test_10x10_singular(self):
        for func in self.pivoted_cholesky:
            # Positive-semidefinite 10x10 matrix A with rank 7
            A = numpy.array([[1.9128, 1.2956, 1.9677, 1.9981, 1.0043, 1.5698, 1.4975, 1.7545, 1.6550, 0.9766],
                            [1.2956, 1.4168, 1.5442, 1.3072, 0.9378, 1.1862, 1.1063, 1.4770, 1.6506, 0.8525],
                            [1.9677, 1.5442, 2.9171, 1.8568, 1.5438, 2.0130, 1.9078, 2.3226, 1.9746, 1.2607],
                            [1.9981, 1.3072, 1.8568, 2.6965, 1.4133, 1.3444, 1.4561, 2.3295, 2.1552, 1.0712],
                            [1.0043, 0.9378, 1.5438, 1.4133, 1.7113, 0.8184, 0.8848, 2.0081, 1.9986, 0.8619],
                            [1.5698, 1.1862, 2.0130, 1.3444, 0.8184, 1.7707, 1.4870, 1.5878, 1.2986, 0.8827],
                            [1.4975, 1.1063, 1.9078, 1.4561, 0.8848, 1.4870, 1.3664, 1.5652, 1.3177, 0.8497],
                            [1.7545, 1.4770, 2.3226, 2.3295, 2.0081, 1.5878, 1.5652, 2.8837, 2.5764, 1.2818],
                            [1.6550, 1.6506, 1.9746, 2.1552, 1.9986, 1.2986, 1.3177, 2.5764, 2.8480, 1.1582],
                            [0.9766, 0.8525, 1.2607, 1.0712, 0.8619, 0.8827, 0.8497, 1.2818, 1.1582, 0.7015]])
            L, piv, rank = func(A, lower=True)
            self.assertEqual(rank, 7)
            # Check if L^T * L == P^T * A * P
            P = numpy.zeros((10, 10))
            P[piv, numpy.arange(10)] = 1
            PtAP = numpy.linalg.multi_dot([P.T, A, P])
            LtL = numpy.dot(L, L.T)
            for i in range(10):
                for j in range(i+1, 10):
                    self.assertEqual(L[i, j], 0)
            self.assertTrue(numpy.allclose(LtL, PtAP, atol=1.0e-12))

    def test_complex(self):
        numpy.random.seed(1)
        A = numpy.random.rand(8,8) + numpy.random.rand(8,8)*1j
        A -= .7 + .3j
        A = A.dot(A.conj().T)
        U, piv = scipy_helper.pivoted_cholesky_python(A)[:2]
        U1 = U.copy()
        U[:,piv] = U1
        self.assertAlmostEqual(abs(U.conj().T.dot(U) - A).max(), 0, 9)


if __name__ == "__main__":
    print("Full tests for scipy_helper")
    unittest.main()
