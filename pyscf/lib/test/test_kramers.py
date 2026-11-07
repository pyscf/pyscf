#!/usr/bin/env python
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

import unittest
import numpy
import scipy.linalg
from pyscf.lib import kramers

BACKENDS = ['numpy']
if kramers.zquatev is not None:
    BACKENDS.append('zquatev')


def _structured(n, seed):
    '''Random Hermitian [[A, B], [-B*, A*]]'''
    rng = numpy.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    a = a + a.conj().T
    b = b - b.T
    return numpy.block([[a, b], [-b.conj(), a.conj()]])


def _time_reversal(c):
    n = c.shape[0] // 2
    return numpy.concatenate((-c[n:].conj(), c[:n].conj()))


class KnownValues(unittest.TestCase):
    def _check(self, h, s, e, c, nvec=None):
        n2 = h.shape[0]
        if s is None:
            s = numpy.eye(n2)
        if nvec is None:
            nvec = n2
        self.assertEqual(c.shape, (n2, nvec))
        self.assertAlmostEqual(abs(e[0::2] - e[1::2]).max(), 0, 14)
        self.assertTrue((numpy.diff(e) > -1e-12).all())
        self.assertAlmostEqual(abs(c.conj().T.dot(s).dot(c) - numpy.eye(nvec)).max(), 0, 12)
        # projected onto the span of the eigenvectors (S may be singular)
        r = c.conj().T.dot(h.dot(c) - s.dot(c) * e)
        self.assertAlmostEqual(abs(r).max() / max(1, abs(e).max()), 0, 12)
        self.assertAlmostEqual(abs(c[:,1::2] - _time_reversal(c[:,0::2])).max(), 0, 14)

    def test_standard(self):
        for n in (1, 2, 3, 40, 77):
            h = _structured(n, n)
            ref = numpy.linalg.eigvalsh(h)
            for backend in BACKENDS:
                e, c = kramers.eigh(h, backend=backend, iop=1)
                self._check(h, None, e, c)
                self.assertAlmostEqual(abs(e - ref).max(), 0, 10)

    def test_block_size(self):
        h = _structured(50, 1)
        e0, c0 = kramers.eigh(h, backend='numpy', iop=1)
        for nb in (1, 7, 64):
            e, c = kramers._eigh_householder(kramers._symmetrize(
                h[kramers._interleave_index(50)[:,None], kramers._interleave_index(50)]), nb)
            self.assertAlmostEqual(abs(e - e0).max(), 0, 11)

    def test_degenerate(self):
        # every eigenvalue four-fold (two Kramers pairs)
        n = 60
        u = kramers.eigh(_structured(n, 3), backend='numpy', iop=1)[1]
        lev = numpy.repeat(numpy.arange(n // 2), 4).astype(float)
        h = (u * lev).dot(u.conj().T)
        for backend in BACKENDS:
            e, c = kramers.eigh(h, backend=backend, iop=1)
            self._check(h, None, e, c)
            self.assertAlmostEqual(abs(e - lev).max(), 0, 11)

    def test_generalized(self):
        n = 50
        h = _structured(n, 4)
        m = _structured(n, 5) / numpy.sqrt(n)
        s = m.dot(m.conj().T) + .1 * numpy.eye(2*n)
        ref = scipy.linalg.eigh(h, s, eigvals_only=True)
        for backend in BACKENDS:
            e, c = kramers.eigh(h, s, backend=backend, iop=1)
            self._check(h, s, e, c)
            self.assertAlmostEqual(abs(e - ref).max(), 0, 9)

    def test_singular_overlap(self):
        # S with a two-pair null space, the linear dependencies removed on
        # request (lindep): canonical orthogonalization
        n = 30
        h = _structured(n, 6)
        u = kramers.eigh(_structured(n, 7), backend='numpy', iop=1)[1]
        w = numpy.repeat(numpy.r_[0, 0, numpy.linspace(.5, 2, n-2)], 2)
        s = (u * w).dot(u.conj().T)
        for backend in BACKENDS:
            e, c = kramers.eigh(h, s, backend=backend, iop=1, lindep=1e-6)
            self._check(h, s, e, c, nvec=2*n-4)
        # lindep has no meaning for the standard problem
        self.assertRaises(ValueError, kramers.eigh, h, lindep=1e-6)

    def test_backends_agree(self):
        if len(BACKENDS) < 2:
            self.skipTest('zquatev not installed')
        h = _structured(40, 8)
        e0, c0 = kramers.eigh(h, backend='numpy', iop=1)
        e1, c1 = kramers.eigh(h, backend='zquatev', iop=1)
        self.assertAlmostEqual(abs(e0 - e1).max(), 0, 11)
        # the same eigenspaces: |<c0_k|c1_k>| of the Kramers pair is 1
        ov = c0.conj().T.dot(c1)
        p = abs(ov[0::2,0::2])**2 + abs(ov[0::2,1::2])**2
        self.assertAlmostEqual(abs(numpy.diag(p) - 1).max(), 0, 9)

    def test_iop(self):
        # iop=0: (u_1..u_n, Tu_1..Tu_n), the same vectors as iop=1
        n = 30
        h = _structured(n, 11)
        for backend in BACKENDS:
            e1, c1 = kramers.eigh(h, backend=backend, iop=1)
            e0, c0 = kramers.eigh(h, backend=backend, iop=0)
            self.assertAlmostEqual(abs(e0 - numpy.hstack((e1[0::2], e1[1::2]))).max(), 0, 14)
            self.assertAlmostEqual(abs(c0 - numpy.hstack((c1[:,0::2], c1[:,1::2]))).max(), 0, 14)
            self.assertAlmostEqual(abs(c0[:,n:] - _time_reversal(c0[:,:n])).max(), 0, 14)
        self.assertRaises(ValueError, kramers.eigh, h, iop=2)

    def test_symmetrization(self):
        # tiny deviations from the structure are removed
        h = _structured(20, 9)
        e0 = kramers.eigh(h, backend='numpy', iop=1)[0]
        e1, c1 = kramers.eigh(h + 1e-10 * _structured(20, 10).real, backend='numpy', iop=1)
        self.assertAlmostEqual(abs(c1[:,1::2] - _time_reversal(c1[:,0::2])).max(), 0, 14)
        self.assertAlmostEqual(abs(e1 - e0).max(), 0, 8)


if __name__ == "__main__":
    print("Full Tests for lib.kramers")
    unittest.main()
