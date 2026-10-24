# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

"""Small-matrix reference tests for periodic linear algebra helpers."""
import contextlib
import io
import unittest

import numpy as np

from pyscf import lib
from pyscf.pbc.lib import arnoldi, linalg_helper as lh


class KnownValues(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.matrix = np.diag(np.arange(1., 9.)) + .03 * rng.normal(size=(8, 8))

    def assert_eigenpairs(self, matrix, values, vectors, atol=1e-6):
        np.testing.assert_allclose(matrix @ vectors, vectors * values,
                                   atol=atol, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), 1., atol=1e-10)

    def test_asymmetric_eigenpairs(self):
        for matrix in (self.matrix, np.array([[1., -2.], [2., 1.]])):
            with self.subTest(shape=matrix.shape):
                values, vectors = lh.diagonalize_asymm(matrix)
                self.assert_eigenpairs(matrix, values, vectors, atol=1e-12)
                self.assertTrue(np.all(np.diff(values.real) >= 0))
                values, left, right = lh.eigGeneral(matrix)
                self.assert_eigenpairs(matrix, values, right, atol=1e-12)
                np.testing.assert_allclose(left.conj().T @ right, np.eye(len(values)),
                                           atol=1e-12)
                np.testing.assert_allclose(matrix.T @ left, left * values.conj(),
                                           atol=1e-12)
                np.testing.assert_allclose((right * values) @ left.conj().T, matrix,
                                           atol=1e-12)
                if np.iscomplexobj(values) and np.any(values.imag):
                    self.assertLess(values[0].imag, values[1].imag)

    def test_svd_cut(self):
        for threshold, rank in ((0., 3), (.1, 2), (10., 0)):
            with self.subTest(threshold=threshold):
                matrix = np.diag([4., 2., .01])
                u, s, vh = lh.svd_cut(matrix, threshold)
                self.assertEqual(len(s), rank)
                if rank:
                    np.testing.assert_allclose(u.T @ u, np.eye(rank), atol=1e-14)
                    np.testing.assert_allclose(vh @ vh.T, np.eye(rank), atol=1e-14)
                    expected = matrix.copy()
                    expected[rank:, rank:] = 0
                    np.testing.assert_allclose((u * s) @ vh, expected, atol=1e-14)
                else:
                    self.assertEqual((u, s, vh), ([], [], []))

    def test_modified_gram_schmidt(self):
        old = np.eye(4)[:1]
        residuals = np.array([[1., 2., 0., 0.], [0., 4., 0., 0.],
                              [2., 0., 3., 0.], [1., 0., 0., 0.]])
        for method in (0, 1):
            with self.subTest(method=method):
                count, vectors = lh.mgs_ortho(old.copy(), residuals.copy(), 1e-12,
                                               iop=method)
                self.assertEqual(count, 2)
                q = np.vstack([old, vectors])
                np.testing.assert_allclose(q @ q.T, np.eye(3), atol=1e-12)
                np.testing.assert_allclose(q.T @ q @ residuals.T, residuals.T,
                                           atol=1e-12)
                count, vectors = lh.mgs_ortho(old, old.copy(), 1e-12, iop=method)
                self.assertEqual(count, 0)
                self.assertEqual(vectors, [])

    def test_real_invariant_subspace(self):
        matrix = np.array([[1., -2., 0.], [2., 1., 0.], [0., 0., 4.]])
        values, left, right = lh.eigGeneral(matrix)
        q = lh.realRepresentation(left, right, 2)
        self.assertFalse(np.iscomplexobj(q))
        np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(q @ q.T @ matrix @ q, matrix @ q, atol=1e-12)

    def test_davidson(self):
        expected = np.sort(np.linalg.eigvals(self.matrix).real)[:2]
        for diagonal in (None, self.matrix.diagonal()):
            for guess in (None, np.eye(8, 2)):
                with self.subTest(diagonal=diagonal is not None, guess=guess is not None):
                    conv, values, vectors, niter = lh.davidson(
                        self.matrix.dot, 8, 2, x0=guess, Adiag=diagonal, verbose=0)
                    np.testing.assert_allclose(values, expected, atol=1e-6, rtol=0)
                    self.assert_eigenpairs(self.matrix, values, vectors)
                    self.assertLess(niter, 8)
        with lib.temporary_env(lh, method='davidson'):
            conv, values, vectors = lh.eigs(self.matrix.dot, 8, 2, verbose=0)
            np.testing.assert_allclose(values, expected, atol=1e-6, rtol=0)
        # A diagonal problem must converge before expanding to the full space.
        diagonal = np.arange(1., 9.)
        conv, values, vectors, niter = lh.davidson(lambda x: diagonal * x, 8, 1, verbose=0)
        self.assertTrue(conv)
        np.testing.assert_allclose(values, [1.])

    def test_davidson_zl(self):
        solver = lh.DavidsonZL()
        solver.ndim = 8
        solver.neig = 2
        solver.matvec = self.matrix.dot
        solver.diag = self.matrix.diagonal()
        solver.iprt = 0
        np.testing.assert_allclose(solver.genMatrix(), self.matrix)
        values, vectors = solver.solve_full()
        self.assert_eigenpairs(self.matrix, values, vectors, atol=1e-12)
        values, vectors = solver.solve_iter()
        np.testing.assert_allclose(values, np.sort(np.linalg.eigvals(self.matrix).real)[:2],
                                   atol=1e-5, rtol=0)
        self.assert_eigenpairs(self.matrix, values, vectors, atol=1e-5)

    def test_davidson_targeted_roots(self):
        # Each root is identified by its largest overlap with a coordinate
        # vector, independently of the arbitrary phase of eigenvectors.
        exact, right = np.linalg.eig(self.matrix)
        expected = exact[np.argmax(abs(right[:2]), axis=1)]
        for diagonal in (None, self.matrix.diagonal()):
            with self.subTest(diagonal=diagonal is not None):
                with contextlib.redirect_stdout(io.StringIO()):
                    values, vectors, count = lh.davidson_guess(
                        self.matrix.dot, 8, 2, Adiag=diagonal)
                np.testing.assert_allclose(values, expected, atol=1e-6, rtol=0)
                self.assert_eigenpairs(self.matrix, values, vectors)
                self.assertGreater(count, 0)
        with lib.temporary_env(lh, method='davidson'):
            with contextlib.redirect_stdout(io.StringIO()):
                conv, values, vectors = lh.eigs(self.matrix.dot, 8, 2, guess=True)
            self.assertTrue(conv)
            self.assert_eigenpairs(self.matrix, values, vectors)

    def test_targeted_root_at_full_subspace(self):
        matrix = np.array([[1., .2], [.2, 2.]])
        with contextlib.redirect_stdout(io.StringIO()):
            values, vectors, count = lh.davidson_guess(matrix.dot, 2, 1)
        np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix)[:1], atol=1e-12)
        self.assert_eigenpairs(matrix, values, vectors, atol=1e-12)
        self.assertEqual(count, 2)

    def test_arnoldi(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        for module, nroots in ((lh, 1), (arnoldi, 1), (lh, 2), (arnoldi, 2)):
            with self.subTest(module=module.__name__, nroots=nroots):
                np.random.seed(5)
                calls = []

                def matvec(vector, args):
                    calls.append(1)
                    self.assertLess(len(calls), 500, 'Arnoldi did not converge')
                    return self.matrix @ vector

                solver = module.Arnoldi(matvec, np.ones((8, 1)), np.ones((8, 1)),
                                        nroots=nroots, tol=1e-9)
                with contextlib.redirect_stdout(io.StringIO()):
                    values, vectors = solver.solve()
                self.assertTrue(solver.converged)
                np.testing.assert_allclose(values, np.sort(np.linalg.eigvals(self.matrix).real)[:nroots],
                                           atol=1e-8, rtol=0)
                self.assert_eigenpairs(self.matrix, values, vectors, atol=1e-8)

    def test_solver_dispatch(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        for method in ('arnoldi', 'davidson_zl'):
            np.random.seed(5)
            with lib.temporary_env(lh, method=method):
                with contextlib.redirect_stdout(io.StringIO()):
                    values, vectors = lh.eigs(self.matrix.dot, 8, 1)
                self.assert_eigenpairs(self.matrix, values, vectors, atol=1e-5)
        with contextlib.redirect_stdout(io.StringIO()):
            values, vectors = arnoldi.davidson_nosymm(self.matrix.dot, 8, 1)
        self.assert_eigenpairs(self.matrix, values, vectors, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
