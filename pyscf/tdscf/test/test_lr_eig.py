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

import sys
import unittest
from unittest import mock

import numpy

from pyscf.lib import logger
from pyscf.tdscf import _lr_eig


class KnownValues(unittest.TestCase):
    def setUp(self):
        a = numpy.diag([1.0, 2.0, 3.0])
        b = numpy.array([[0.0, 0.05, 0.0], [0.05, 0.0, 0.02], [0.0, 0.02, 0.0]])
        self.matrix = numpy.block([[a, b], [-b, -a]])
        self.hdiag = numpy.r_[numpy.diag(a), -numpy.diag(a)]
        self.guess = numpy.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    def aop(self, x):
        return numpy.asarray(x).dot(self.matrix.T)

    def precond(self, dx, energy):
        denominator = self.hdiag[None, :] - numpy.asarray(energy)[:, None]
        denominator[abs(denominator) < 1e-8] = 1e-8
        return dx / denominator

    def solve(self):
        return _lr_eig.real_eig(
            self.aop,
            self.guess,
            self.precond,
            nroots=1,
            tol_residual=1e-9,
            max_cycle=30,
            verbose=logger.Logger(sys.stdout, logger.QUIET),
        )

    def test_real_eig_restarts_after_singular_subspace(self):
        subspace_solver = _lr_eig.TDDFT_subspace_eigen_solver
        failed = False

        def fail_full_subspace_once(*args, **kwargs):
            nonlocal failed
            if args[0].shape[0] == 3 and not failed:
                failed = True
                raise numpy.linalg.LinAlgError('singular subspace metric')
            return subspace_solver(*args, **kwargs)

        with mock.patch.object(_lr_eig, 'TDDFT_subspace_eigen_solver', fail_full_subspace_once):
            converged, energy, _ = self.solve()

        self.assertTrue(failed)
        self.assertTrue(converged[0])
        self.assertAlmostEqual(energy[0], 0.9991663794740591, 12)

    def test_real_eig_restarts_when_residual_basis_is_dependent(self):
        orthogonalize = _lr_eig.VW_Gram_Schmidt_fill_holder
        empty_calls = 0

        def empty_expanded_basis_once(*args, **kwargs):
            nonlocal empty_calls
            if args[0].shape[1] == 2 and empty_calls < 2:
                empty_calls += 1
                size = args[2].shape[0]
                return numpy.zeros((0, size)), numpy.zeros((0, size))
            return orthogonalize(*args, **kwargs)

        with mock.patch.object(_lr_eig, 'VW_Gram_Schmidt_fill_holder', empty_expanded_basis_once):
            converged, energy, _ = self.solve()

        self.assertEqual(empty_calls, 2)
        self.assertTrue(converged[0])
        self.assertAlmostEqual(energy[0], 0.9991663794740591, 12)

    def test_real_eig_uses_raw_residual_when_preconditioner_returns_zero(self):
        def zero_precond(dx, energy):
            return numpy.zeros_like(dx)

        converged, energy, _ = _lr_eig.real_eig(
            self.aop,
            self.guess,
            zero_precond,
            nroots=1,
            tol_residual=1e-9,
            max_cycle=30,
            verbose=logger.Logger(sys.stdout, logger.QUIET),
        )

        self.assertTrue(converged[0])
        self.assertAlmostEqual(energy[0], 0.9991663794740591, 12)


if __name__ == '__main__':
    unittest.main()
