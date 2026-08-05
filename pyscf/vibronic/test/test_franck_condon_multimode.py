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

'''Multidimensional Franck-Condon overlaps with a genuine Duschinsky rotation.

The decisive check is direct multidimensional Gauss-Hermite quadrature of

    <v_f|v_i> = int dQ psi_f(J Q + K) psi_i(Q)

with a real rotation ``J``, different frequencies in every mode and a nonzero
``K``.  The quadrature shares no algebra with the Doktorov recursion.
'''

import itertools
import math
import unittest

import numpy

from pyscf.vibronic import franck_condon as fc


# ----------------------------------------------------------------------
# Independent quadrature reference
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


def _grid(freq_i, nquad):
    '''Cached tensor Gauss-Hermite grid in the initial-state coordinates.'''
    wi = numpy.asarray(freq_i, dtype=float)
    ni = wi.size
    key = (tuple(wi), nquad)
    cached = _grid._cache.get(key)
    if cached is not None:
        return cached
    x, wt = numpy.polynomial.hermite.hermgauss(nquad)
    grids = numpy.meshgrid(*[x] * ni, indexing='ij')
    weight = numpy.ones_like(grids[0])
    for a in numpy.meshgrid(*[wt] * ni, indexing='ij'):
        weight = weight * a
    scale = numpy.sqrt(2.0 / wi)
    Q = numpy.stack([grids[k] * scale[k] for k in range(ni)], axis=-1)
    jac = float(numpy.prod(scale))
    cached = (Q, weight, jac)
    _grid._cache[key] = cached
    return cached


_grid._cache = {}


def overlap_quadrature(v, w, freq_i, freq_f, J, K, nquad=60):
    '''Gauss-Hermite reference for <v_f|w_i>.

    The initial-state ground Gaussian is absorbed into the Gauss-Hermite
    weight by the substitution ``Q_k = sqrt(2/omega_i,k) x_k``; everything else
    (the Hermite polynomials of both states and the final-state Gaussian) is
    evaluated explicitly on the grid.
    '''
    wi = numpy.asarray(freq_i, dtype=float)
    wf = numpy.asarray(freq_f, dtype=float)
    J = numpy.asarray(J, dtype=float)
    K = numpy.asarray(K, dtype=float)
    Q, weight, jac = _grid(wi, nquad)
    y = (Q.dot(J.T) + K) * numpy.sqrt(wf)
    z = Q * numpy.sqrt(wi)
    norm = float(numpy.prod((wf / numpy.pi) ** 0.25) * numpy.prod((wi / numpy.pi) ** 0.25))
    integrand = (norm * _hermite_product(v, y) * _hermite_product(w, z)
                 * numpy.exp(-0.5 * numpy.sum(y ** 2, axis=-1)))
    return float(numpy.sum(weight * integrand) * jac)


def rot2(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return numpy.array([[c, -s], [s, c]])


def rot3(a_deg, b_deg, c_deg):
    '''Proper rotation R_z(a) R_y(b) R_z(c).'''
    def rz(t):
        co, si = math.cos(t), math.sin(t)
        return numpy.array([[co, -si, 0.0], [si, co, 0.0], [0.0, 0.0, 1.0]])

    def ry(t):
        co, si = math.cos(t), math.sin(t)
        return numpy.array([[co, 0.0, si], [0.0, 1.0, 0.0], [-si, 0.0, co]])
    return rz(math.radians(a_deg)).dot(ry(math.radians(b_deg))).dot(rz(math.radians(c_deg)))


# Reference systems used throughout
FREQ_I2 = numpy.array([0.0080, 0.0150])
FREQ_F2 = numpy.array([0.0110, 0.0190])
K2 = numpy.array([0.55, -0.35])

FREQ_I3 = numpy.array([0.0060, 0.0120, 0.0210])
FREQ_F3 = numpy.array([0.0090, 0.0140, 0.0260])
K3 = numpy.array([0.40, -0.60, 0.25])


class KnownValues(unittest.TestCase):

    # -- the decisive quadrature comparisons ---------------------------
    def test_2d_rotation_25deg_vs_quadrature(self):
        J = rot2(25.0)
        worst = 0.0
        for v in itertools.product(range(5), repeat=2):
            got = fc.multimode_overlaps(FREQ_I2, FREQ_F2, J, K2, [list(v)])[0]
            ref = overlap_quadrature(v, (0, 0), FREQ_I2, FREQ_F2, J, K2, nquad=90)
            worst = max(worst, abs(got - ref))
        # measured worst absolute error 4.9e-16
        self.assertLess(worst, 1e-9)

    def test_2d_rotation_45deg_strong_mixing(self):
        J = rot2(45.0)
        worst = 0.0
        for v in itertools.product(range(5), repeat=2):
            got = fc.multimode_overlaps(FREQ_I2, FREQ_F2, J, K2, [list(v)])[0]
            ref = overlap_quadrature(v, (0, 0), FREQ_I2, FREQ_F2, J, K2, nquad=90)
            worst = max(worst, abs(got - ref))
        # measured worst absolute error 5.6e-16
        self.assertLess(worst, 1e-9)

    def test_2d_hot_bands_vs_quadrature(self):
        '''General <v_f|v_i> (the finite-temperature kernel) against quadrature.'''
        J = rot2(45.0)
        worst = 0.0
        for v in itertools.product(range(3), repeat=2):
            for w in itertools.product(range(3), repeat=2):
                got = fc.multimode_overlaps(FREQ_I2, FREQ_F2, J, K2, [list(v)], [list(w)])[0]
                ref = overlap_quadrature(v, w, FREQ_I2, FREQ_F2, J, K2, nquad=90)
                worst = max(worst, abs(got - ref))
        # measured worst absolute error 5.6e-16
        self.assertLess(worst, 1e-9)

    def test_3d_rotation_vs_quadrature(self):
        J = rot3(40.0, 25.0, -15.0)
        worst = 0.0
        for v in itertools.product(range(3), repeat=3):
            got = fc.multimode_overlaps(FREQ_I3, FREQ_F3, J, K3, [list(v)])[0]
            ref = overlap_quadrature(v, (0, 0, 0), FREQ_I3, FREQ_F3, J, K3, nquad=50)
            worst = max(worst, abs(got - ref))
        # measured worst absolute error 2.2e-12 with nquad=50; it drops to
        # 2.2e-16 at nquad=70, confirming the residual is quadrature error
        # and not the recursion.
        self.assertLess(worst, 1e-9)

    def test_2d_nonorthogonal_j(self):
        '''A slightly non-orthogonal J (mode-space leakage) is still handled.'''
        J = rot2(30.0).dot(numpy.diag([1.02, 0.97]))
        J[0, 1] += 0.05
        worst = 0.0
        for v in itertools.product(range(4), repeat=2):
            got = fc.multimode_overlaps(FREQ_I2, FREQ_F2, J, K2, [list(v)])[0]
            ref = overlap_quadrature(v, (0, 0), FREQ_I2, FREQ_F2, J, K2, nquad=90)
            worst = max(worst, abs(got - ref))
        # measured worst absolute error 3.3e-16
        self.assertLess(worst, 1e-9)

    def test_nonorthogonal_j_needs_general_prefactor(self):
        '''det J != 1 requires the GENERAL determinant form of <0_f|0_i>.

        REFERENCES.md section 5.2: the compact Huh prefactor
        ``2^(n/2) |det R|^(1/2)`` (Eq. 5.11) silently assumes ``|det J| = 1``.
        For a deliberately non-orthogonal J it is wrong by several percent.
        This test pins down the general form (Eq. 5.9) by (i) comparing to
        quadrature and (ii) asserting that the compact form is measurably
        *different*, so a regression to Eq. 5.11 cannot pass silently.
        '''
        J = numpy.array([[1.20, -0.35], [0.42, 1.05]])
        detJ = numpy.linalg.det(J)
        self.assertGreater(abs(abs(detJ) - 1.0), 0.2)        # measured det J = 1.407

        got = fc.overlap_00(FREQ_I2, FREQ_F2, J, K2)
        ref = overlap_quadrature((0, 0), (0, 0), FREQ_I2, FREQ_F2, J, K2, nquad=120)
        self.assertAlmostEqual(got, ref, 12)

        # The rejected compact form, built from the same Doktorov matrices.
        dok = fc._Doktorov(FREQ_I2, FREQ_F2, J, K2)
        Jbar = numpy.sqrt(FREQ_F2)[:, None] * J / numpy.sqrt(FREQ_I2)[None, :]
        Qref = numpy.linalg.inv(numpy.eye(2) + Jbar.T.dot(Jbar))
        Rref = Qref.dot(Jbar.T)
        compact = (2.0 ** (2 / 2.0) * math.sqrt(abs(numpy.linalg.det(Rref)))
                   * math.exp(-0.5 * dok.d.dot(dok.d - dok.Pff.dot(dok.d))))
        self.assertGreater(abs(compact - ref) / ref, 0.05)   # measured 19% error

        # ... while the general form agrees with quadrature for the whole
        # progression, not just the origin.
        worst = 0.0
        for v in itertools.product(range(4), repeat=2):
            a = fc.multimode_overlaps(FREQ_I2, FREQ_F2, J, K2, [list(v)])[0]
            b = overlap_quadrature(v, (0, 0), FREQ_I2, FREQ_F2, J, K2, nquad=120)
            worst = max(worst, abs(a - b))
        self.assertLess(worst, 1e-9)

    def test_doktorov_projector_identities(self):
        '''Pi = [[Pii, Pfi^T], [Pfi, Pff]] must be an orthogonal projector.

        Equivalently W = 1 - 2*Pi is self-inverse (REFERENCES.md Eqs. 5.6-5.7).
        This is the internal consistency check that catches the ``P = 2 J Q J^T``
        transcription error described in REFERENCES.md section 5.5(1).
        '''
        for freq_i, freq_f, J, K in (
                (FREQ_I2, FREQ_F2, rot2(45.0), K2),
                (FREQ_I3, FREQ_F3, rot3(40.0, 25.0, -15.0), K3),
                (FREQ_I2, FREQ_F2, numpy.array([[1.20, -0.35], [0.42, 1.05]]), K2)):
            dok = fc._Doktorov(freq_i, freq_f, J, K)
            n = len(freq_i)
            Pi = numpy.block([[dok.Pii, dok.Pfi.T], [dok.Pfi, dok.Pff]])
            self.assertAlmostEqual(float(numpy.abs(Pi.dot(Pi) - Pi).max()), 0.0, 12)
            W = numpy.eye(2 * n) - 2.0 * Pi
            self.assertAlmostEqual(float(numpy.abs(W.dot(W) - numpy.eye(2 * n)).max()),
                                   0.0, 12)
            self.assertAlmostEqual(float(numpy.abs(W - W.T).max()), 0.0, 14)
            # eigenvalues of 1 - 2P lie in (-1, 1]: the reason the recursion is bounded
            ev = numpy.linalg.eigvalsh(dok.Rff)
            self.assertTrue(numpy.all(ev >= -1.0 - 1e-12) and numpy.all(ev <= 1.0 + 1e-12))

    def test_equal_frequency_displaced_00_is_not_unity(self):
        '''Guards against the P = 2*J*Q*J^T error (REFERENCES.md 5.5(1)).

        With omega_f = omega_i and J = I, that erroneous form gives
        <0|0> = 1 for ANY displacement instead of exp(-S/2).
        '''
        freq = numpy.array([0.011, 0.017])
        # measured (K, S_total, <0|0>, exp(-S/2)):
        #   [ 1.3, -0.8]  0.01474  0.992659573499  0.992659573499
        #   [ 8.0, -6.0]  0.65800  0.719643016747  0.719643016747
        #   [15.0,-12.0]  2.46150  0.292073440434  0.292073440434
        #   [25.0,-20.0]  6.83750  0.032753351050  0.032753351050
        for K in ([1.3, -0.8], [8.0, -6.0], [15.0, -12.0], [25.0, -20.0]):
            K = numpy.array(K)
            S = float(fc.huang_rhys(freq, K).sum())
            got = fc.overlap_00(freq, freq, numpy.eye(2), K)
            self.assertAlmostEqual(got, math.exp(-0.5 * S), 14)
        # the discriminating case: S = 6.84 gives <0|0> = 0.0328, whereas the
        # erroneous P = 2*J*Q*J^T form would return 1.0 here.
        self.assertLess(fc.overlap_00(freq, freq, numpy.eye(2),
                                      numpy.array([25.0, -20.0])), 0.05)

    # -- limits --------------------------------------------------------
    def test_identity_j_is_product_of_1d(self):
        n = 3
        J = numpy.eye(n)
        worst = 0.0
        for v in itertools.product(range(4), repeat=n):
            got = fc.multimode_overlaps(FREQ_I3, FREQ_F3, J, K3, [list(v)])[0]
            ref = 1.0
            for k in range(n):
                ref *= fc.overlap_1d(v[k], 0, FREQ_I3[k], FREQ_F3[k], K3[k])
            worst = max(worst, abs(got - ref))
        # measured worst absolute error 2.8e-17
        self.assertLess(worst, 1e-13)

    def test_kronecker_delta_limit(self):
        '''J = I, K = 0, omega_f = omega_i  =>  <v_f|0_i> = delta_{v,0}.'''
        freq = numpy.array([0.007, 0.013, 0.022])
        J = numpy.eye(3)
        K = numpy.zeros(3)
        states, _ = fc.enumerate_states(3, 4)
        ov = fc.multimode_overlaps(freq, freq, J, K, states)
        ground = numpy.all(states == 0, axis=1)
        self.assertAlmostEqual(ov[ground][0], 1.0, 14)
        self.assertAlmostEqual(float(numpy.abs(ov[~ground]).max()), 0.0, 14)

    def test_kronecker_delta_general_states(self):
        '''Same limit for hot bands: <v_f|w_i> = delta_{v,w}.'''
        freq = numpy.array([0.007, 0.013])
        J = numpy.eye(2)
        K = numpy.zeros(2)
        pairs = list(itertools.product(itertools.product(range(3), repeat=2), repeat=2))
        v = numpy.array([p[0] for p in pairs])
        w = numpy.array([p[1] for p in pairs])
        ov = fc.multimode_overlaps(freq, freq, J, K, v, w)
        ref = numpy.array([1.0 if tuple(a) == tuple(b) else 0.0 for a, b in zip(v, w)])
        self.assertAlmostEqual(float(numpy.abs(ov - ref).max()), 0.0, 13)

    # -- sum rule ------------------------------------------------------
    def test_sum_rule_converges_2d(self):
        '''The sum-rule deficit is genuine TRUNCATION, not algorithm error.

        A finite ``max_quanta`` can never give an exact sum rule: the deficit
        ``1 - sum_v |<v_f|0_i>|^2`` is exactly the intensity carried by the
        states that were not enumerated.  So the meaningful assertion is that
        the deficit decreases monotonically and geometrically towards zero,
        which is what is checked here (rather than pretending some particular
        ``max_quanta`` is "exact").

        Measured 2-D convergence table (J = 30 deg, freq/K from FREQ_I2 etc.):

            max_quanta   n_states   sum_rule            deficit
                     1          3   0.959814987308      4.019e-02
                     2          6   0.997502574222      2.497e-03
                     4         15   0.999836320971      1.637e-04
                     8         45   0.999999200614      7.994e-07
                    16        153   0.999999999977      2.317e-11
                    30        496   1.000000000000     -4.441e-16
        '''
        J = rot2(30.0)
        quanta = (1, 2, 4, 8, 16, 30)
        vals = []
        for mq in quanta:
            res = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2, max_quanta=mq,
                                           verbose=0)
            vals.append(res.sum_rule)
        deficits = [1.0 - v for v in vals]

        # (i) the sum rule is a probability: never above 1, never below 0
        for v in vals:
            self.assertLessEqual(v, 1.0 + 1e-12)
            self.assertGreaterEqual(v, 0.0)
        # (ii) monotonically non-decreasing in max_quanta
        for a, b in zip(vals, vals[1:]):
            self.assertGreaterEqual(b, a - 1e-14)
        # (iii) the deficit really shrinks, by >= 3 orders of magnitude
        #       between max_quanta = 2 and 16
        self.assertGreater(deficits[1] / max(deficits[4], 1e-300), 1e3)
        # (iv) a severely truncated space is visibly incomplete
        self.assertLess(vals[0], 0.97)
        # (v) at max_quanta = 30 the residual is below double precision noise
        self.assertLess(abs(deficits[-1]), 1e-12)

    def test_sum_rule_converges_3d(self):
        '''Same statement in 3-D: the deficit is truncation and it decreases.

        Measured 3-D convergence table (J = rot3(40, 25, -15)):

            max_quanta   n_states   sum_rule            deficit
                     4         35   0.998425848063      1.574e-03
                     8        165   0.999966067978      3.393e-05
                    14        680   0.999999877681      1.223e-07
                    20       1771   0.999999999528      4.725e-10
                    26       3654   0.999999999998      1.892e-12
        '''
        J = rot3(40.0, 25.0, -15.0)
        quanta = (4, 8, 14, 20, 26)
        vals = []
        nstates = []
        for mq in quanta:
            res = fc.franck_condon_factors(FREQ_I3, FREQ_F3, J, K3, max_quanta=mq,
                                           verbose=0)
            vals.append(res.sum_rule)
            nstates.append(res.nstate)
        deficits = [1.0 - v for v in vals]

        self.assertEqual(nstates, [35, 165, 680, 1771, 3654])
        for a, b in zip(vals, vals[1:]):
            self.assertGreaterEqual(b, a - 1e-14)
        # every refinement gains at least one order of magnitude
        for a, b in zip(deficits, deficits[1:]):
            self.assertLess(b, 0.1 * a)
        # documented physically-justified bound at max_quanta = 20
        self.assertLess(deficits[3], 1e-8)
        # and at max_quanta = 26 the space is converged to ~1e-12
        self.assertLess(deficits[4], 1e-10)

    # -- invariances ---------------------------------------------------
    def test_column_sign_flip_invariance(self):
        '''Flipping column j of J is a phase change of initial mode j.

        ``psi_0^(i)`` is even in every coordinate, so with ``v_i = 0`` every
        overlap is *exactly* unchanged.  For a hot band the overlap picks up
        ``(-1)**w_j``.
        '''
        J = rot3(40.0, 25.0, -15.0)
        for j in range(3):
            S = numpy.eye(3)
            S[j, j] = -1.0
            Jp = J.dot(S)
            states, _ = fc.enumerate_states(3, 3)
            a = fc.multimode_overlaps(FREQ_I3, FREQ_F3, J, K3, states)
            b = fc.multimode_overlaps(FREQ_I3, FREQ_F3, Jp, K3, states)
            self.assertAlmostEqual(float(numpy.abs(a - b).max()), 0.0, 14)
            # hot band: sign factor (-1)**w_j
            w = numpy.zeros((states.shape[0], 3), dtype=int)
            w[:, j] = 1
            a1 = fc.multimode_overlaps(FREQ_I3, FREQ_F3, J, K3, states, w)
            b1 = fc.multimode_overlaps(FREQ_I3, FREQ_F3, Jp, K3, states, w)
            self.assertAlmostEqual(float(numpy.abs(a1 + b1).max()), 0.0, 14)

    def test_row_sign_flip_invariance(self):
        '''Flipping row k of J together with K[k] multiplies <v|0> by (-1)**v_k.

        This is a phase change of *final* mode k: ``Q_f -> T Q_f`` with
        ``T = diag(1,...,-1,...,1)``, under which ``psi_v^(f)`` picks up
        ``(-1)**v_k``.  |<v_f|0>|^2 is therefore always unchanged; the sign of
        the overlap flips exactly when ``v_k`` is odd (note: v_k, not the total
        number of quanta).
        '''
        J = rot3(40.0, 25.0, -15.0)
        states, _ = fc.enumerate_states(3, 4)
        base = fc.multimode_overlaps(FREQ_I3, FREQ_F3, J, K3, states)
        for k in range(3):
            T = numpy.eye(3)
            T[k, k] = -1.0
            flipped = fc.multimode_overlaps(FREQ_I3, FREQ_F3, T.dot(J), T.dot(K3), states)
            sign = numpy.where(states[:, k] % 2 == 0, 1.0, -1.0)
            self.assertAlmostEqual(float(numpy.abs(flipped - sign * base).max()), 0.0, 14)
            self.assertAlmostEqual(float(numpy.abs(flipped ** 2 - base ** 2).max()), 0.0, 14)

    def test_degenerate_subspace_invariance(self):
        '''An orthogonal re-mixing of two exactly degenerate final modes leaves
        the *summed* intensity over the degenerate manifold invariant.'''
        freq_i = numpy.array([0.0075, 0.0130, 0.0185])
        freq_f = numpy.array([0.0110, 0.0110, 0.0240])     # modes 0 and 1 degenerate
        J = rot3(35.0, 20.0, 10.0)
        K = numpy.array([0.5, -0.3, 0.45])

        theta = math.radians(37.0)
        c, s = math.cos(theta), math.sin(theta)
        U = numpy.eye(3)
        U[0, 0] = c
        U[0, 1] = -s
        U[1, 0] = s
        U[1, 1] = c

        states, _ = fc.enumerate_states(3, 5)
        a = fc.multimode_overlaps(freq_i, freq_f, J, K, states) ** 2
        b = fc.multimode_overlaps(freq_i, freq_f, U.dot(J), U.dot(K), states) ** 2

        # group by (v0 + v1, v2): the degenerate manifold
        groups = {}
        for n in range(states.shape[0]):
            key = (int(states[n, 0]) + int(states[n, 1]), int(states[n, 2]))
            groups.setdefault(key, []).append(n)

        n_changed = 0
        worst_group = 0.0
        for key, idx in groups.items():
            sa = float(a[idx].sum())
            sb = float(b[idx].sum())
            worst_group = max(worst_group, abs(sa - sb))
            if max(abs(a[i] - b[i]) for i in idx) > 1e-8:
                n_changed += 1
        # individual FCFs really do change ...
        self.assertGreater(n_changed, 0)
        # ... but the summed intensity in each degenerate manifold does not.
        # measured worst deviation 3e-17
        self.assertLess(worst_group, 1e-12)
        # and the total is of course conserved
        self.assertAlmostEqual(float(a.sum()), float(b.sum()), 12)

    def test_reciprocity_multimode(self):
        '''Q_f = J Q_i + K  <=>  Q_i = J^T Q_f - J^T K (J orthogonal).'''
        J = rot3(40.0, 25.0, -15.0)
        Jt = J.T
        Kt = -J.T.dot(K3)
        for v in itertools.product(range(3), repeat=3):
            for w in ((0, 0, 0), (1, 0, 0), (0, 2, 1)):
                a = fc.multimode_overlaps(FREQ_I3, FREQ_F3, J, K3, [list(v)], [list(w)])[0]
                b = fc.multimode_overlaps(FREQ_F3, FREQ_I3, Jt, Kt, [list(w)], [list(v)])[0]
                self.assertAlmostEqual(a, b, 12)

    # -- shape / input validation --------------------------------------
    def test_shape_errors(self):
        J = rot2(25.0)
        self.assertRaises(ValueError, fc.multimode_overlaps,
                          FREQ_I2, FREQ_F2, numpy.eye(3), K2, [[0, 0]])
        self.assertRaises(ValueError, fc.multimode_overlaps,
                          FREQ_I2, FREQ_F2, J, [1.0], [[0, 0]])
        self.assertRaises(ValueError, fc.multimode_overlaps,
                          FREQ_I2, FREQ_F2, J, K2, [[0, 0, 0]])
        self.assertRaises(ValueError, fc.multimode_overlaps,
                          FREQ_I2, FREQ_F2, J, K2, [[-1, 0]])
        self.assertRaises(ValueError, fc.multimode_overlaps,
                          FREQ_I2, FREQ_F2, J, K2, [[0, 0], [1, 0]], [[0, 0], [0, 0], [0, 0]])

    def test_doktorov_metric_cannot_be_made_singular_by_j(self):
        '''The Duschinsky metric is structurally non-singular -- and why.

        ``A = J^T Omega_f J + Omega_i`` is a sum of a positive semi-definite
        matrix and a positive definite one, so ``A >= Omega_i`` and its
        smallest eigenvalue is bounded below by ``min(freq_i)``, whatever J is.
        In the reduced variables this is the statement ``1 + Jbar^T Jbar >= 1``.
        **There is therefore no J -- orthogonal or not, however badly
        conditioned -- that can make the metric singular**, so a "singular J"
        test would be vacuous.  What *is* asserted here:

        1. a zero or negative initial frequency is rejected up front
           (``ValueError``), because that is the only real way to lose
           positive-definiteness;
        2. even a wildly non-orthogonal, nearly rank-deficient J leaves A well
           conditioned and the overlaps correct against quadrature;
        3. the ``cond_tol`` guard fires only for an extreme *frequency* spread,
           i.e. cond(A) ~ max(freq_f)/min(freq_i).
        '''
        # (1) non-positive initial frequency -> ValueError before any algebra
        self.assertRaises(ValueError, fc.overlap_00,
                          [1e-30, 0.01], FREQ_F2, numpy.eye(2), K2)
        self.assertRaises(ValueError, fc.overlap_00,
                          [-0.01, 0.01], FREQ_F2, numpy.eye(2), K2)

        # (2) a nearly rank-deficient J (sigma_min = 1e-8) is still fine
        J = numpy.array([[1.0, 1.0], [1.0, 1.0 + 1e-8]])
        self.assertLess(numpy.linalg.svd(J, compute_uv=False)[-1], 1e-8)
        dok = fc._Doktorov(FREQ_I2, FREQ_F2, J, K2)
        A = J.T.dot(numpy.diag(FREQ_F2)).dot(J) + numpy.diag(FREQ_I2)
        # smallest eigenvalue of A is bounded below by min(freq_i)
        self.assertGreaterEqual(float(numpy.linalg.eigvalsh(A)[0]), FREQ_I2.min() - 1e-18)
        self.assertLess(numpy.linalg.cond(A), 1e4)
        got = dok.amplitudes([[0, 0], [1, 0], [0, 1], [2, 1]])
        ref = [overlap_quadrature(v, (0, 0), FREQ_I2, FREQ_F2, J, K2, nquad=120)
               for v in ((0, 0), (1, 0), (0, 1), (2, 1))]
        self.assertAlmostEqual(float(numpy.abs(got - numpy.array(ref)).max()), 0.0, 12)

        # (3) the cond_tol guard fires on an extreme frequency spread only
        self.assertRaises(RuntimeError, fc.overlap_00,
                          [1e-11, 100.0], [1e-11, 100.0], numpy.eye(2), [0.0, 0.0])

    def test_large_displacement_is_finite_and_reported(self):
        '''Extreme displacement: finite numbers, and the loss is *reported*.

        This replaces a singular-metric test (see
        ``test_doktorov_metric_cannot_be_made_singular_by_j``) with the failure
        mode that can actually occur.  As K grows at fixed ``max_quanta`` the
        intensity migrates to high quanta, so the sum rule must degrade -- and
        that degradation must show up in ``result.sum_rule``, never as NaN and
        never as a silently plausible-looking number.
        '''
        J = rot2(25.0)
        states, _ = fc.enumerate_states(2, 6)
        sums = []
        for scale in (1.0, 10.0, 50.0, 100.0, 200.0, 500.0):
            K = K2 * scale
            ov = fc.multimode_overlaps(FREQ_I2, FREQ_F2, J, K, states)
            self.assertTrue(numpy.all(numpy.isfinite(ov)),
                            'non-finite overlap at K scale %g' % scale)
            # an overlap is an inner product of normalised states
            self.assertTrue(numpy.all(numpy.abs(ov) <= 1.0 + 1e-12))
            res = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K, max_quanta=6,
                                           verbose=0)
            self.assertTrue(numpy.isfinite(res.sum_rule))
            self.assertLessEqual(res.sum_rule, 1.0 + 1e-12)
            sums.append(res.sum_rule)
        # measured (K scale, total Huang-Rhys S, sum_rule at max_quanta = 6):
        #     1     0.0028   0.999993929701
        #    10     0.2828   0.999963808012
        #    50     7.0688   0.432040369406
        #   100    28.2750   5.67645698387e-07
        #   200   113.1000   8.22759750685e-40
        #   500   706.8750   1.21472378150e-289
        for a, b in zip(sums, sums[1:]):
            self.assertLessEqual(b, a + 1e-12)
        self.assertGreater(sums[0], 0.99)
        self.assertLess(sums[2], 0.5)          # S = 7 already loses most intensity
        self.assertLess(sums[4], 1e-30)        # S = 113 loses essentially all of it
        # the honest summary must flag the collapse rather than hide it
        res = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2 * 50.0, max_quanta=6,
                                       verbose=0)
        self.assertIn('WARNING', res.summary())
        self.assertLess(res.sum_rule, 0.5)

    def test_extreme_displacement_underflow_is_zero_not_nan(self):
        '''At S ~ 1e4 the origin overlap underflows; it must give 0.0, not NaN.'''
        J = rot2(25.0)
        K = numpy.array([2000.0, -1500.0])
        states, _ = fc.enumerate_states(2, 4)
        ov = fc.multimode_overlaps(FREQ_I2, FREQ_F2, J, K, states)
        self.assertTrue(numpy.all(numpy.isfinite(ov)))
        self.assertTrue(numpy.all(ov == 0.0))
        # but the log-space prefactor is still informative
        dok = fc._Doktorov(FREQ_I2, FREQ_F2, J, K)
        self.assertTrue(numpy.isfinite(dok.log_overlap_00))
        self.assertLess(dok.log_overlap_00, -700.0)

    def test_broad_frequency_spread_is_finite(self):
        freq_i = numpy.array([1e-5, 1.0])
        freq_f = numpy.array([2e-5, 0.5])
        J = rot2(20.0)
        K = numpy.array([0.1, 0.02])
        ov = fc.multimode_overlaps(freq_i, freq_f, J, K, [[0, 0], [1, 1], [2, 0]])
        self.assertTrue(numpy.all(numpy.isfinite(ov)))


class EnumerationTests(unittest.TestCase):

    def test_counts_dense(self):
        '''With no restrictions the enumeration is the full simplex.'''
        for nmode in (1, 3, 4):
            for mq in (0, 1, 2, 3):
                states, info = fc.enumerate_states(nmode, mq)
                ref = fc._binom(nmode + mq, mq)
                self.assertEqual(states.shape, (ref, nmode))
                self.assertEqual(info['n_enumerated'], ref)
                self.assertEqual(info['n_kept'], ref)
                self.assertFalse(info['truncated'])
                self.assertTrue(numpy.all(states.sum(axis=1) <= mq))
                # no duplicates
                self.assertEqual(len(set(map(tuple, states.tolist()))), ref)

    def test_ordering_is_deterministic(self):
        a, _ = fc.enumerate_states(4, 3)
        b, _ = fc.enumerate_states(4, 3)
        self.assertTrue(numpy.array_equal(a, b))
        self.assertEqual(a.tobytes(), b.tobytes())
        # sorted by (total quanta, occupation vector lexicographically)
        keys = [(int(r.sum()), tuple(int(x) for x in r)) for r in a]
        self.assertEqual(keys, sorted(keys))

    def test_dtype_and_shape(self):
        states, _ = fc.enumerate_states(5, 2)
        self.assertEqual(states.dtype, numpy.int16)
        self.assertEqual(states.shape[1], 5)
        states, _ = fc.enumerate_states(0, 3)
        self.assertEqual(states.shape, (1, 0))

    def test_class_1_ignores_active_modes(self):
        '''Class 1 is always exhaustive over ALL modes.'''
        states, info = fc.enumerate_states(4, 2, active_modes=[0, 1])
        singles = states[(states > 0).sum(axis=1) == 1]
        excited = sorted(set(int(numpy.argmax(r)) for r in singles))
        self.assertEqual(excited, [0, 1, 2, 3])
        # class 2 only among the active modes
        doubles = states[(states > 0).sum(axis=1) == 2]
        for r in doubles:
            self.assertEqual(sorted(numpy.where(r > 0)[0].tolist()), [0, 1])
        # 1 + (4 singles at t=1) + (4 singles at t=2) + 1 double = 10
        self.assertEqual(info['n_enumerated'], 10)
        self.assertEqual(states.shape[0], 10)

    def test_max_modes_excited(self):
        states, info = fc.enumerate_states(4, 4, max_modes_excited=2)
        self.assertTrue(numpy.all((states > 0).sum(axis=1) <= 2))
        self.assertEqual(info['max_modes_excited'], 2)

    def test_max_quanta_per_mode(self):
        states, info = fc.enumerate_states(3, 5, max_quanta_per_mode=2)
        self.assertTrue(numpy.all(states <= 2))
        self.assertEqual(info['max_quanta_per_mode'], 2)

    def test_truncation_is_reported(self):
        full, full_info = fc.enumerate_states(5, 4)
        cut, info = fc.enumerate_states(5, 4, max_states=20)
        self.assertEqual(cut.shape[0], 20)
        self.assertTrue(info['truncated'])
        self.assertEqual(info['n_enumerated'], full_info['n_enumerated'])
        self.assertEqual(info['n_kept'], 20)
        self.assertEqual(info['n_skipped'], full_info['n_enumerated'] - 20)
        # truncation keeps the first states in the deterministic order
        self.assertTrue(numpy.array_equal(cut, full[:20]))

    def test_hard_limit_raises(self):
        self.assertRaises(ValueError, fc.enumerate_states, 30, 8, hard_limit=1000)
        # with max_states the same request succeeds and reports the truncation
        states, info = fc.enumerate_states(30, 8, max_states=500, hard_limit=1000)
        self.assertEqual(states.shape[0], 500)
        self.assertTrue(info['truncated'])

    def test_invalid_arguments(self):
        self.assertRaises(ValueError, fc.enumerate_states, -1, 2)
        self.assertRaises(ValueError, fc.enumerate_states, 3, -1)
        self.assertRaises(ValueError, fc.enumerate_states, 3, 2, active_modes=[5])


class DriverTests(unittest.TestCase):

    def test_energies_and_e00(self):
        J = rot2(25.0)
        e_ad = 0.15
        res = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2, e_adiabatic=e_ad,
                                       max_quanta=3)
        zpe_i = 0.5 * FREQ_I2.sum()
        zpe_f = 0.5 * FREQ_F2.sum()
        self.assertAlmostEqual(res.e_00, e_ad + zpe_f - zpe_i, 14)
        ref = res.e_00 + res.states.astype(float).dot(FREQ_F2)
        self.assertAlmostEqual(float(numpy.abs(res.energies - ref).max()), 0.0, 14)
        self.assertAlmostEqual(float(numpy.abs(res.fcf - res.overlaps ** 2).max()), 0.0, 15)
        self.assertTrue(numpy.all(res.populations == 1.0))
        self.assertTrue(numpy.all(res.init_states == 0))
        self.assertIsNone(res.duschinsky)
        self.assertIn('sum rule', res.summary())
        self.assertIn('FranckCondonResult', repr(res))

    def test_intensity_pruning_bookkeeping(self):
        J = rot2(25.0)
        full = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2, max_quanta=6)
        pruned = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2, max_quanta=6,
                                          intensity_threshold=1e-6)
        self.assertLess(pruned.nstate, full.nstate)
        # sum rule is computed BEFORE pruning, so it is unchanged
        self.assertAlmostEqual(pruned.sum_rule, full.sum_rule, 15)
        self.assertEqual(pruned.truncation['n_pruned'], full.nstate - pruned.nstate)
        self.assertTrue(numpy.all(pruned.fcf >= 1e-6))

    def test_active_threshold_selection(self):
        freq_i = numpy.array([0.008, 0.015, 0.02])
        freq_f = numpy.array([0.011, 0.019, 0.021])
        J = numpy.eye(3)
        K = numpy.array([0.9, 1e-6, 1e-6])          # only mode 0 is displaced
        res = fc.franck_condon_factors(freq_i, freq_f, J, K, max_quanta=3,
                                       active_threshold=1e-4, j_offdiag_threshold=0.1)
        doubles = res.states[(res.states > 0).sum(axis=1) >= 2]
        self.assertEqual(doubles.shape[0], 0)       # no other mode is active
        # class 1 is still exhaustive
        singles = res.states[(res.states > 0).sum(axis=1) == 1]
        self.assertEqual(sorted(set(int(numpy.argmax(r)) for r in singles)), [0, 1, 2])

    def test_determinism(self):
        J = rot3(40.0, 25.0, -15.0)
        a = fc.franck_condon_factors(FREQ_I3, FREQ_F3, J, K3, max_quanta=5)
        b = fc.franck_condon_factors(FREQ_I3, FREQ_F3, J, K3, max_quanta=5)
        self.assertEqual(a.states.tobytes(), b.states.tobytes())
        self.assertEqual(a.overlaps.tobytes(), b.overlaps.tobytes())

    def test_finite_temperature_matches_quadrature(self):
        '''T > 0 uses the general <v_f|v_i> kernel; check it against quadrature.'''
        J = rot2(35.0)
        T = 4000.0                                    # hot enough to populate v=1
        res = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2, max_quanta=2,
                                       temperature=T, max_quanta_init=2,
                                       population_threshold=1e-6, verbose=0)
        self.assertGreater(int((res.init_states.sum(axis=1) > 0).sum()), 0)
        worst = 0.0
        for n in range(res.nstate):
            v = tuple(int(x) for x in res.states[n])
            w = tuple(int(x) for x in res.init_states[n])
            ref = overlap_quadrature(v, w, FREQ_I2, FREQ_F2, J, K2, nquad=90)
            worst = max(worst, abs(res.overlaps[n] - ref))
        # measured worst absolute error 5.6e-16
        self.assertLess(worst, 1e-9)

    def test_finite_temperature_energies_and_populations(self):
        J = rot2(35.0)
        T = 3000.0
        res = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2, max_quanta=3,
                                       temperature=T, max_quanta_init=3, verbose=0)
        ref = (res.e_00 + res.states.astype(float).dot(FREQ_F2)
               - res.init_states.astype(float).dot(FREQ_I2))
        self.assertAlmostEqual(float(numpy.abs(res.energies - ref).max()), 0.0, 14)
        # populations are the exact harmonic Boltzmann weights
        kt = 3000.0 * 3.1668115634556e-06
        ground = numpy.where(res.init_states.sum(axis=1) == 0)[0][0]
        x = numpy.exp(-FREQ_I2 / kt)
        p0 = float(numpy.prod(1.0 - x))
        self.assertAlmostEqual(res.populations[ground], p0, 6)
        # hot bands appear below the origin
        self.assertLess(float(res.energies.min()), res.e_00)

    def test_finite_temperature_sum_rule(self):
        '''At T > 0 the sum rule is sum_w p_w sum_v |<v|w>|^2 -> 1.'''
        J = rot2(30.0)
        res = fc.franck_condon_factors(FREQ_I2, FREQ_F2, J, K2, max_quanta=20,
                                       temperature=2000.0, max_quanta_init=8,
                                       population_threshold=1e-10, verbose=0)
        # measured sum_rule 0.999984307516 with an initial-state population
        # coverage of 0.999984334555 -- i.e. the deficit is entirely the
        # truncation of the thermal initial-state list, not the recursion.
        captured = res.truncation['init_population_captured']
        self.assertAlmostEqual(res.sum_rule, 1.0, 4)
        self.assertAlmostEqual(res.sum_rule / captured, 1.0, 7)


if __name__ == '__main__':
    print('Full Tests for pyscf.vibronic.franck_condon (multimode)')
    unittest.main()
