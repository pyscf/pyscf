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

'''One-dimensional harmonic Franck-Condon overlaps.

Every reference value here is either a closed-form analytic result or an
independent Gauss-Hermite quadrature of the defining integral.
'''

import math
import unittest

import numpy

from pyscf.vibronic import franck_condon as fc


# ----------------------------------------------------------------------
# Independent reference: direct Gauss-Hermite quadrature of
#     <v_f|w_i> = int dQ psi_f(J Q + K) psi_i(Q)
# built only from the *definition* of the harmonic eigenfunctions.  It shares
# no algebra with the Doktorov implementation under test.
# ----------------------------------------------------------------------

def _hermite_phys(n, x):
    '''Physicists' Hermite polynomial H_n(x) by the standard recursion.'''
    h0 = numpy.ones_like(x)
    if n == 0:
        return h0
    h1 = 2.0 * x
    for k in range(1, n):
        h0, h1 = h1, 2.0 * x * h1 - 2.0 * k * h0
    return h1


def _hermite_product(occ, coord):
    out = numpy.ones(coord.shape[:-1])
    for k, nk in enumerate(occ):
        out = out * _hermite_phys(nk, coord[..., k]) / math.sqrt(2.0 ** nk * math.factorial(nk))
    return out


def overlap_quadrature(v, w, freq_i, freq_f, J, K, nquad=100):
    '''Gauss-Hermite reference for <v_f|w_i>.

    The initial-state ground Gaussian exp(-omega_i Q^2/2) is used as the
    quadrature weight through the substitution Q_k = sqrt(2/omega_i,k) x_k,
    which turns it into the Gauss-Hermite weight exp(-x^2) exactly.
    '''
    wi = numpy.asarray(freq_i, dtype=float)
    wf = numpy.asarray(freq_f, dtype=float)
    J = numpy.asarray(J, dtype=float)
    K = numpy.asarray(K, dtype=float)
    ni = wi.size
    x, wt = numpy.polynomial.hermite.hermgauss(nquad)
    grids = numpy.meshgrid(*[x] * ni, indexing='ij')
    weight = numpy.ones_like(grids[0])
    for a in numpy.meshgrid(*[wt] * ni, indexing='ij'):
        weight = weight * a
    scale = numpy.sqrt(2.0 / wi)
    Q = numpy.stack([grids[k] * scale[k] for k in range(ni)], axis=-1)
    jac = float(numpy.prod(scale))
    y = (Q.dot(J.T) + K) * numpy.sqrt(wf)
    z = Q * numpy.sqrt(wi)
    norm = float(numpy.prod((wf / numpy.pi) ** 0.25) * numpy.prod((wi / numpy.pi) ** 0.25))
    integrand = (norm * _hermite_product(v, y) * _hermite_product(w, z)
                 * numpy.exp(-0.5 * numpy.sum(y ** 2, axis=-1)))
    return float(numpy.sum(weight * integrand) * jac)


class KnownValues(unittest.TestCase):

    def test_poisson_progression(self):
        '''Equal frequencies, pure displacement: |<n|0>|^2 = exp(-S) S^n / n!.'''
        omega = 0.0123
        worst = 0.0
        for disp in (0.3, 1.0, 2.5, 5.0):
            S = fc.huang_rhys(omega, disp)
            tab = fc.overlap_1d_table(20, 0, omega, omega, disp)
            for n in range(21):
                ref = math.exp(-S) * S ** n / math.factorial(n)
                got = tab[n, 0] ** 2
                err = abs(got - ref) / max(ref, 1e-300)
                worst = max(worst, err)
                self.assertAlmostEqual(got / ref, 1.0, 12)
        # measured worst relative error ~1.5e-15
        self.assertLess(worst, 1e-12)

    def test_huang_rhys_definition(self):
        self.assertAlmostEqual(fc.huang_rhys(0.02, 3.0), 0.5 * 0.02 * 9.0, 15)
        S = fc.huang_rhys(numpy.array([0.01, 0.02]), numpy.array([1.0, 2.0]))
        self.assertAlmostEqual(S[0], 0.005, 15)
        self.assertAlmostEqual(S[1], 0.04, 15)

    def test_frequency_change_zero_displacement(self):
        '''Odd overlaps vanish; <0|0> and <2|0> have known closed forms.'''
        wi, wf = 0.009, 0.017
        tab = fc.overlap_1d_table(9, 0, wi, wf, 0.0)
        ref00 = math.sqrt(2.0 * math.sqrt(wi * wf) / (wi + wf))
        self.assertAlmostEqual(tab[0, 0], ref00, 14)
        for n in range(1, 10, 2):
            self.assertAlmostEqual(tab[n, 0], 0.0, 15)
        # <2|0> = <0|0> * (w_f - w_i) / (sqrt(2) (w_f + w_i))
        ref02 = ref00 * (wf - wi) / (math.sqrt(2.0) * (wf + wi))
        self.assertAlmostEqual(tab[2, 0], ref02, 14)
        # and against quadrature
        q = overlap_quadrature((2,), (0,), [wi], [wf], numpy.eye(1), [0.0], nquad=120)
        self.assertAlmostEqual(tab[2, 0], q, 12)

    def test_general_1d_vs_quadrature(self):
        '''Frequency change AND displacement, vs Gauss-Hermite quadrature.'''
        wi, wf = 0.0104, 0.0071
        disp = -0.83
        J = numpy.eye(1)
        worst = 0.0
        for n in range(9):
            got = fc.overlap_1d(n, 0, wi, wf, disp)
            ref = overlap_quadrature((n,), (0,), [wi], [wf], J, [disp], nquad=140)
            worst = max(worst, abs(got - ref))
            self.assertAlmostEqual(got, ref, 10)
        # measured worst absolute error ~1e-15
        self.assertLess(worst, 1e-10)

    def test_hot_band_1d_vs_quadrature(self):
        '''Full <n_f|n_i> table against quadrature (finite-temperature kernel).'''
        wi, wf = 0.0088, 0.0135
        disp = 0.62
        tab = fc.overlap_1d_table(5, 5, wi, wf, disp)
        worst = 0.0
        for nf in range(6):
            for ni in range(6):
                ref = overlap_quadrature((nf,), (ni,), [wi], [wf],
                                              numpy.eye(1), [disp], nquad=160)
                worst = max(worst, abs(tab[nf, ni] - ref))
                self.assertAlmostEqual(tab[nf, ni], ref, 10)
        self.assertLess(worst, 1e-10)

    def test_table_matches_scalar(self):
        wi, wf, disp = 0.011, 0.019, 1.4
        tab = fc.overlap_1d_table(6, 4, wi, wf, disp)
        for nf in (0, 3, 6):
            for ni in (0, 2, 4):
                self.assertAlmostEqual(tab[nf, ni],
                                       fc.overlap_1d(nf, ni, wi, wf, disp), 14)

    def test_sum_rule_1d(self):
        '''sum_n |<n|0>|^2 -> 1 with enough quanta.'''
        for wi, wf, disp in ((0.008, 0.014, 1.1), (0.02, 0.005, -0.4), (0.01, 0.01, 3.0)):
            tab = fc.overlap_1d_table(300, 0, wi, wf, disp)
            s = float((tab[:, 0] ** 2).sum())
            self.assertAlmostEqual(s, 1.0, 10)

    def test_sum_rule_hot_band(self):
        '''sum_{n_f} |<n_f|n_i>|^2 = 1 for each fixed n_i (completeness).'''
        tab = fc.overlap_1d_table(400, 4, 0.0092, 0.0151, 0.7)
        for ni in range(5):
            self.assertAlmostEqual(float((tab[:, ni] ** 2).sum()), 1.0, 9)

    def test_reciprocity(self):
        '''Q_f = Q_i + D  <=>  Q_i = Q_f - D, and the overlap is symmetric.

        Both wavefunctions are real, so <n_f|n_i> = <n_i|n_f>.  Swapping the
        two oscillators means (w_i, w_f, D) -> (w_f, w_i, -D).
        '''
        wi, wf, disp = 0.0074, 0.0163, 1.25
        for nf in range(5):
            for ni in range(5):
                a = fc.overlap_1d(nf, ni, wi, wf, disp)
                b = fc.overlap_1d(ni, nf, wf, wi, -disp)
                self.assertAlmostEqual(a, b, 13)

    def test_identity_limit(self):
        '''Same frequency, no displacement -> Kronecker delta.'''
        tab = fc.overlap_1d_table(6, 6, 0.013, 0.013, 0.0)
        self.assertTrue(numpy.allclose(tab, numpy.eye(7), atol=1e-13))

    def test_multimode_1d_agrees_with_overlap_1d(self):
        wi, wf, disp = numpy.array([0.0121]), numpy.array([0.0087]), numpy.array([0.95])
        states = numpy.arange(8).reshape(-1, 1)
        got = fc.multimode_overlaps(wi, wf, numpy.eye(1), disp, states)
        ref = numpy.array([fc.overlap_1d(n, 0, wi[0], wf[0], disp[0]) for n in range(8)])
        self.assertAlmostEqual(float(abs(got - ref).max()), 0.0, 14)

    def test_overlap_00_closed_form(self):
        wi, wf = 0.0091, 0.0224
        got = fc.overlap_00([wi], [wf], numpy.eye(1), [0.0])
        self.assertAlmostEqual(got, math.sqrt(2.0 * math.sqrt(wi * wf) / (wi + wf)), 14)
        self.assertIs(fc.overlap_0_0, fc.overlap_00)

    def test_negative_or_zero_frequency_raises(self):
        self.assertRaises(ValueError, fc.overlap_00, [0.0], [0.01], numpy.eye(1), [0.0])
        self.assertRaises(ValueError, fc.overlap_00, [0.01], [-0.01], numpy.eye(1), [0.0])
        self.assertRaises(ValueError, fc.overlap_1d, 0, 0, 0.0, 0.01, 0.0)

    def test_bad_quantum_numbers(self):
        self.assertRaises(ValueError, fc.overlap_1d, -1, 0, 0.01, 0.01, 0.0)
        self.assertRaises(ValueError, fc.overlap_1d_table, 2, -3, 0.01, 0.01, 0.0)

    def test_large_displacement_finite(self):
        '''A huge displacement must underflow gracefully, never to NaN/inf.'''
        tab = fc.overlap_1d_table(40, 0, 0.01, 0.01, 60.0)
        self.assertTrue(numpy.all(numpy.isfinite(tab)))
        # S = 800 -> <0|0> = exp(-400) = 1.9e-174, still representable
        self.assertAlmostEqual(fc.overlap_00([0.01], [0.01], numpy.eye(1), [400.0]),
                               math.exp(-400.0), 180)
        # S = 20000 -> underflows; must return exactly 0.0, never NaN
        self.assertEqual(fc.overlap_00([0.01], [0.01], numpy.eye(1), [2000.0]), 0.0)
        dok = fc._Doktorov([0.01], [0.01], numpy.eye(1), [2000.0])
        self.assertTrue(numpy.isfinite(dok.log_overlap_00))
        self.assertAlmostEqual(dok.log_overlap_00, -10000.0, 8)


if __name__ == '__main__':
    print('Full Tests for pyscf.vibronic.franck_condon (1-D)')
    unittest.main()
