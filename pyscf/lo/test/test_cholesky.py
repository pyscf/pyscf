#!/usr/bin/env python
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


import unittest
import numpy
from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.lo.cholesky import cholesky_mos, _pivoted_cholesky_python


mol = Mole()
mol.atom = '''
C        0.681068338      0.605116159      0.307300799
C       -0.733665805      0.654940451     -0.299036438
C       -1.523996730     -0.592207689      0.138683275
H        0.609941801      0.564304456      1.384183068
H        1.228991034      1.489024155      0.015946420
H       -1.242251083      1.542928348      0.046243898
H       -0.662968178      0.676527364     -1.376503770
H       -0.838473936     -1.344174292      0.500629028
H       -2.075136399     -0.983173387     -0.703807608
H       -2.212637905     -0.323898759      0.926200671
O        1.368219958     -0.565620846     -0.173113101
H        2.250134219     -0.596689848      0.204857736
'''
mol.basis = 'STO-3G'
mol.verbose = 0
mol.output = None
mol.build()
mf = RHF(mol)
mf.conv_tol = 1.0e-12
mf.kernel()


def compare_arrays(obj, A, B, delta):
    difference_norm = numpy.linalg.norm(A - B)
    obj.assertAlmostEqual(difference_norm, 0.0, delta=delta)


def compare_arrays_strict(obj, A, B):
    obj.assertEqual(numpy.array_equal(A, B), True)


class KnownValues(unittest.TestCase):

    def setUp(self):
        self.mol = mol.copy()
        self.mo_coeff = mf.mo_coeff.copy()
        self.nocc = numpy.count_nonzero(mf.mo_occ > 0)
        self.rdm1_rhf = mf.make_rdm1()
        self.sao = mf.get_ovlp()
    
    def test_density(self):
        '''
        Test whether the localized orbitals preserve the density.
        '''
        # density from the localized orbitals
        mo_loc = cholesky_mos(self.mo_coeff[:, :self.nocc])
        rdm_loc = 2 * mo_loc.dot(mo_loc.T)

        # check the norm of the difference
        compare_arrays(self, self.rdm1_rhf, rdm_loc, 1.0e-12)
    
    def test_orth(self):
        '''
        Test whether the localized orbitals are orthonormal.
        '''
        # localized virtual MOs
        mo_loc = cholesky_mos(self.mo_coeff[:, :self.nocc])

        # test LMO overlap integral
        smo = numpy.linalg.multi_dot([mo_loc.T, self.sao, mo_loc])
        compare_arrays(self, smo, numpy.eye(self.nocc), 1.0e-12)
    
    def test_localization(self):
        '''
        Check a few selected values of the orbital coefficient matrix.
        '''
        mo_loc = cholesky_mos(self.mo_coeff[:, :self.nocc])
        delta = 1.0e-6
        self.assertAlmostEqual(abs(mo_loc[22, 0]), 1.02618438, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[6, 3]), 0.10412481, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[27, 5]), 0.17253633, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[6, 8]), 0.63599723, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[14, 11]), 0.5673705, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[4, 15]), 0.51124407, delta=delta)


class PythonCholesky(unittest.TestCase):

    def test_1x1(self):
        A = numpy.array([[9.0]])
        L, piv, rank = _pivoted_cholesky_python(A)
        self.assertEqual(L.shape, (1, 1))
        self.assertEqual(piv.shape, (1,))
        self.assertAlmostEqual(L[0, 0], 3.0, delta=1.0e-14)
        self.assertEqual(piv[0], 0)
        self.assertEqual(rank, 1)
    
    def test_2x2(self):
        A = numpy.array([[9.0, 6.0], [6.0, 5.0]])
        L, piv, rank = _pivoted_cholesky_python(A)
        L_ref = numpy.array([[3.0, 0.0], [2.0, 1.0]])
        piv_ref = numpy.array([0, 1])
        compare_arrays(self, L, L_ref, delta=1.0e-14)
        compare_arrays_strict(self, piv, piv_ref)
        self.assertEqual(rank, 2)
    
    def test_2x2_singular(self):
        A = numpy.array([[1.0, 2.0], [2.0, 4.0]])
        L, piv, rank = _pivoted_cholesky_python(A, tol=1.0e-14)
        L_ref = numpy.array([[2.0, 0.0], [1.0, 0.0]])
        piv_ref = numpy.array([1, 0])
        compare_arrays(self, L, L_ref, delta=1.0e-14)
        compare_arrays_strict(self, piv, piv_ref)
        self.assertEqual(rank, 1)

    def test_10x10(self):
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
        L, piv, rank = _pivoted_cholesky_python(A, tol=1.0e-12)
        self.assertEqual(rank, 10)
        # Check if L^T * L == P^T * A * P
        P = numpy.zeros((10, 10))
        P[piv, numpy.arange(10)] = 1
        PtAP = numpy.linalg.multi_dot([P.T, A, P])
        LtL = numpy.dot(L, L.T)
        for i in range(10):
            for j in range(i+1, 10):
                self.assertEqual(L[i, j], 0)
        compare_arrays(self, LtL, PtAP, delta=1.0e-12)

    def test_10x10_singular(self):
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
        L, piv, rank = _pivoted_cholesky_python(A, tol=1.0e-12)
        self.assertEqual(rank, 7)
        # Check if L^T * L == P^T * A * P
        P = numpy.zeros((10, 10))
        P[piv, numpy.arange(10)] = 1
        PtAP = numpy.linalg.multi_dot([P.T, A, P])
        LtL = numpy.dot(L, L.T)
        for i in range(10):
            for j in range(i+1, 10):
                self.assertEqual(L[i, j], 0)
        compare_arrays(self, LtL, PtAP, delta=1.0e-12)


if __name__ == "__main__":
    print("Test Cholesky localization")
    unittest.main()
