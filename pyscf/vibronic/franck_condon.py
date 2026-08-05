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

r'''
Harmonic Franck-Condon overlap kernels (Doktorov recursion).

This module contains the numerical heart of :mod:`pyscf.vibronic`.  It works
exclusively on plain NumPy arrays -- ``freq_i``, ``freq_f``, ``J``, ``K`` -- so
it can be tested (and used) without any electronic-structure input.

Conventions (see :mod:`pyscf.vibronic.units`, all Hartree atomic units,
:math:`\hbar = 1`)::

    Q_f = J Q_i + K

with ``Q`` mass-weighted normal coordinates in bohr*sqrt(m_e), ``J`` of shape
``(n_f, n_i)`` (**rows = final-state modes, columns = initial-state modes**),
``K`` of shape ``(n_f,)`` in bohr*sqrt(m_e), and ``freq`` the angular
frequencies :math:`\omega` in Eh.

Derivation
==========

Write :math:`\Omega_i = \mathrm{diag}(\omega_i)`,
:math:`\Omega_f = \mathrm{diag}(\omega_f)`.  The normalised harmonic
eigenfunctions are

.. math::

    \psi^{(i)}_w(Q) &= N_i \prod_j \frac{H_{w_j}(z_j)}{\sqrt{2^{w_j} w_j!}}
                       e^{-z^T z/2}, \qquad z = \Omega_i^{1/2} Q, \\
    \psi^{(f)}_v(Q_f) &= N_f \prod_k \frac{H_{v_k}(y_k)}{\sqrt{2^{v_k} v_k!}}
                       e^{-y^T y/2}, \qquad y = \Omega_f^{1/2} Q_f,

with :math:`N_i = \prod_j (\omega_{i,j}/\pi)^{1/4}` and likewise for
:math:`N_f`.  Substituting :math:`Q_f = J Q + K` gives :math:`y = C Q + d` with

.. math::   C = \Omega_f^{1/2} J, \qquad d = \Omega_f^{1/2} K .

The overlap is :math:`\langle v_f|w_i\rangle = \int \mathrm{d}Q\,
\psi^{(f)}_v(JQ+K)\, \psi^{(i)}_w(Q)` (the Jacobian is unity for an orthogonal,
square ``J``; for a rectangular ``J`` the integral over ``Q_i`` is taken as the
definition).

**Origin overlap.**  For :math:`v = w = 0` the integrand is a pure Gaussian
:math:`\exp(-\tfrac12 Q^T A Q - b^T Q - \tfrac12 d^T d)` with

.. math::   A = J^T \Omega_f J + \Omega_i, \qquad b = C^T d = J^T \Omega_f K,

so that

.. math::

    \langle 0_f | 0_i \rangle =
        \frac{(2\pi)^{n_i/2}}{\pi^{(n_i+n_f)/4}}
        \Big(\prod_j \omega_{i,j} \prod_k \omega_{f,k}\Big)^{1/4}
        (\det A)^{-1/2}
        \exp\!\Big(-\tfrac12 d^T (1 - P) d\Big),

where :math:`P_{ff} \equiv P = C A^{-1} C^T`.  ``A`` is symmetric positive
definite whenever every :math:`\omega_i > 0`, so the origin overlap is always
real and *positive*.  In one dimension with :math:`J=1, K=0` this collapses to
the familiar :math:`\sqrt{2\sqrt{\omega_i\omega_f}/(\omega_i+\omega_f)}`.

**Recursion.**  Introduce the generating function

.. math::

    F(t, s) = \sum_{v, w} \prod_k \frac{t_k^{v_k}}{\sqrt{v_k!}}
                          \prod_j \frac{s_j^{w_j}}{\sqrt{w_j!}}
                          \langle v_f | w_i \rangle .

Using :math:`\sum_n H_n(x)\sigma^n/n! = e^{2\sigma x - \sigma^2}` with
:math:`\sigma = t/\sqrt 2` the sums exponentiate, the remaining ``Q`` integral
is Gaussian, and one obtains the closed form

.. math::

    F(t,s) = \langle 0_f|0_i\rangle
        \exp\!\Big(\delta^T t + \epsilon^T s
                   + \tfrac12 t^T R_{ff} t + \tfrac12 s^T R_{ii} s
                   + t^T R_{fi} s \Big)

with (writing :math:`D = \Omega_i^{1/2}`)

.. math::

    P_{ff} &= C A^{-1} C^T, \quad P_{fi} = C A^{-1} D^T, \quad
    P_{ii} = D A^{-1} D^T, \\
    R_{ff} &= 2 P_{ff} - 1, \quad R_{ii} = 2 P_{ii} - 1, \quad
    R_{fi} = 2 P_{fi}, \\
    \delta &= \sqrt 2\,(1 - P_{ff})\, d, \qquad
    \epsilon = -\sqrt 2\, P_{fi}^T d .

Matching powers in :math:`\partial F/\partial t_k` and
:math:`\partial F/\partial s_k` yields the Doktorov recursions

.. math::

    \sqrt{v_k}\,\langle v|w\rangle &=
        \delta_k \langle v - e_k|w\rangle
        + \sum_j (R_{ff})_{kj} \sqrt{(v-e_k)_j}\,\langle v - e_k - e_j|w\rangle
        + \sum_j (R_{fi})_{kj} \sqrt{w_j}\, \langle v - e_k|w - e_j\rangle, \\
    \sqrt{w_k}\,\langle v|w\rangle &=
        \epsilon_k \langle v|w - e_k\rangle
        + \sum_j (R_{ii})_{kj} \sqrt{(w-e_k)_j}\,\langle v|w - e_k - e_j\rangle
        + \sum_j (R_{fi})_{jk} \sqrt{v_j}\, \langle v - e_j|w - e_k\rangle .

(E. V. Doktorov, I. A. Malkin, V. I. Man'ko, *J. Mol. Spectrosc.* **64**, 302
(1977).)  Every factor above has been validated against direct Gauss-Hermite
quadrature in ``pyscf/vibronic/test/test_franck_condon_multimode.py``.

Correspondence with ``work/franck_condon/REFERENCES.md``
========================================================

The reference document works in the *reduced* variables
:math:`\mathbb{J} = \Omega_f^{1/2} J \Omega_i^{-1/2}`,
:math:`\delta = \Omega_f^{1/2} K` and defines
:math:`Q_{\rm ref} = (1 + \mathbb{J}^T\mathbb{J})^{-1}`,
:math:`R_{\rm ref} = Q_{\rm ref}\mathbb{J}^T`,
:math:`P_{\rm ref} = \mathbb{J} Q_{\rm ref} \mathbb{J}^T`.  Since
:math:`1 + \mathbb{J}^T\mathbb{J} = \Omega_i^{-1/2} A\, \Omega_i^{-1/2}`, the
unreduced quantities above map onto them exactly:

===============================  ==========================================
This module                      REFERENCES.md
===============================  ==========================================
``Pff`` :math:`= C A^{-1} C^T`   :math:`P = \mathbb{J} Q \mathbb{J}^T`  (5.5)
``Pii`` :math:`= D A^{-1} D^T`   :math:`Q = (1+\mathbb{J}^T\mathbb{J})^{-1}` (5.3)
``Pfi.T``                        :math:`R = Q\mathbb{J}^T`  (5.4)
``Rff`` :math:`= 2P_{ff} - 1`    :math:`-A = -(1 - 2P)`  (5.12)
``delta`` :math:`= \sqrt2(1-P)d`  :math:`b`  (5.13)
``d``                            :math:`\delta`  (5.2)
===============================  ==========================================

Two documented pitfalls from REFERENCES.md section 5.5 are avoided here:

* **No spurious factor of 2 in** :math:`P`.  ``Pff`` is :math:`C A^{-1} C^T`,
  which equals :math:`\mathbb{J} Q \mathbb{J}^T` with *no* internal factor of
  two.  The projector property is therefore exact:
  :math:`\Pi = \begin{pmatrix} P_{ii} & P_{fi}^T \\ P_{fi} & P_{ff}\end{pmatrix}`
  satisfies :math:`\Pi^2 = \Pi`, equivalently :math:`W = 1 - 2\Pi` is
  self-inverse.  ``test_doktorov_projector_identities`` asserts this.  The
  ``P = 2 J Q J^T`` variant would give :math:`\langle 0|0\rangle = 1` for any
  displacement at equal frequencies, which the Poisson test rules out.
* **The general determinant form of** :math:`\langle 0_f|0_i\rangle`
  (REFERENCES.md Eq. 5.9) is used, *not* the compact
  :math:`2^{n/2}|\det R|^{1/2}` form (Eq. 5.11).  The compact form silently
  assumes :math:`|\det J| = 1` and is wrong by several percent for a
  non-orthogonal ``J``.  ``test_nonorthogonal_j_needs_general_prefactor``
  covers exactly this case.

Known failure modes
===================

* **Large displacement.**  ``<0_f|0_i>`` decays as ``exp(-S)``; for a total
  Huang-Rhys factor beyond ~700 the origin overlap underflows to zero.  The
  prefactor is therefore evaluated through :func:`numpy.linalg.slogdet` and the
  log is exposed as :attr:`_Doktorov.log_overlap_00`, but the recursion itself
  runs on plain floats and will return ``0.0`` in that regime.  A large
  displacement also moves the intensity maximum to high quanta, so
  ``max_quanta`` must grow with ``S`` or the sum rule collapses.
* **High quanta.**  The recursion adds terms of alternating sign; for
  :math:`\sum_k v_k \gtrsim 60` with a strongly mixing ``J`` catastrophic
  cancellation limits the accuracy to roughly ``1e-10`` relative.
* **Near-singular** :math:`A = J^T\Omega_f J + \Omega_i`.  This can only happen
  if some initial frequency is (numerically) zero or the frequency spread is
  extreme.  The condition number is checked and a :class:`RuntimeError` is
  raised rather than returning garbage.
'''

import itertools
import math
import sys

import numpy
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger
from pyscf.vibronic import units

__all__ = [
    'huang_rhys',
    'overlap_1d', 'overlap_1d_table',
    'overlap_00', 'overlap_0_0',
    'multimode_overlaps',
    'enumerate_states',
    'franck_condon_factors',
    'FranckCondonResult',
]

#: Frequencies at or below this value (Eh) are rejected as unphysical.
FREQ_MIN = 1e-12

#: ``A = J^T Omega_f J + Omega_i`` is rejected above this condition number.
COND_TOL = 1e12

_SQRT2 = math.sqrt(2.0)


def huang_rhys(omega, displacement):
    r'''Dimensionless Huang-Rhys factor :math:`S = \tfrac12 \omega \Delta^2`.

    Args:
        omega : float or (n,) array
            Angular frequency in atomic units (Eh, hbar = 1).
        displacement : float or (n,) array
            Equilibrium displacement along the corresponding normal coordinate
            in bohr*sqrt(m_e), i.e. an element of ``K``.

    Returns:
        float or (n,) ndarray, the dimensionless Huang-Rhys factor.

    With :math:`\bar K = \sqrt{\omega}\,\Delta` the dimensionless displacement,
    :math:`S = \tfrac12 \bar K^2`.  For a displaced oscillator of *unchanged*
    frequency the vibrational progression is Poissonian,
    :math:`|\langle n|0\rangle|^2 = e^{-S} S^n / n!`; this identity is verified
    numerically in the test suite.
    '''
    omega = numpy.asarray(omega, dtype=float)
    displacement = numpy.asarray(displacement, dtype=float)
    return 0.5 * omega * displacement ** 2


def _check_freq(freq, name):
    freq = numpy.asarray(freq, dtype=float).ravel()
    if freq.size and not numpy.all(numpy.isfinite(freq)):
        raise ValueError('%s contains non-finite entries' % name)
    if numpy.any(freq <= FREQ_MIN):
        raise ValueError(
            '%s contains a non-positive or numerically zero frequency '
            '(min = %.6e Eh).  Imaginary or zero modes must be removed or '
            'handled before the Franck-Condon step.' % (name, freq.min() if freq.size else 0.0))
    return freq


class _Doktorov(object):
    '''Pre-computed Doktorov matrices for a given ``(freq_i, freq_f, J, K)``.

    Attributes (see the module docstring for the definitions):
        ``delta`` (n_f,), ``eps`` (n_i,), ``Rff`` (n_f,n_f), ``Rii`` (n_i,n_i),
        ``Rfi`` (n_f,n_i), ``overlap_00`` float, ``log_overlap_00`` float.
    '''

    def __init__(self, freq_i, freq_f, J, K, cond_tol=COND_TOL):
        freq_i = _check_freq(freq_i, 'freq_i')
        freq_f = _check_freq(freq_f, 'freq_f')
        n_i = freq_i.size
        n_f = freq_f.size

        J = numpy.asarray(J, dtype=float)
        if J.shape != (n_f, n_i):
            raise ValueError(
                'J has shape %s but (n_f, n_i) = (%d, %d).  Rows of J index '
                'FINAL-state modes and columns index INITIAL-state modes.'
                % (J.shape, n_f, n_i))
        K = numpy.asarray(K, dtype=float).ravel()
        if K.shape != (n_f,):
            raise ValueError('K has shape %s but (n_f,) = (%d,)' % (K.shape, n_f))
        if not (numpy.all(numpy.isfinite(J)) and numpy.all(numpy.isfinite(K))):
            raise ValueError('J or K contains non-finite entries')

        self.freq_i = freq_i
        self.freq_f = freq_f
        self.J = J
        self.K = K
        self.n_i = n_i
        self.n_f = n_f

        sqrt_wi = numpy.sqrt(freq_i)
        sqrt_wf = numpy.sqrt(freq_f)

        # C = Omega_f^{1/2} J   (n_f, n_i);   D = Omega_i^{1/2}  (diagonal)
        C = sqrt_wf[:, None] * J
        d = sqrt_wf * K                                    # dimensionless displacement K_bar

        A = C.T.dot(C) + numpy.diag(freq_i)                # (n_i, n_i), SPD
        A = 0.5 * (A + A.T)

        if n_i:
            cond = numpy.linalg.cond(A)
            if not numpy.isfinite(cond) or cond > cond_tol:
                raise RuntimeError(
                    'The Duschinsky metric A = J^T Omega_f J + Omega_i is '
                    'numerically singular (condition number %.3e > %.1e).  '
                    'This usually means a (near-)zero initial-state frequency '
                    'or an extreme frequency spread.  Franck-Condon factors '
                    'cannot be evaluated reliably.' % (cond, cond_tol))
            sign, logdet_A = numpy.linalg.slogdet(A)
            if sign <= 0:
                raise RuntimeError('A = J^T Omega_f J + Omega_i is not positive definite')
            cho = scipy.linalg.cho_factor(A, lower=True)
            Ainv_Ct = scipy.linalg.cho_solve(cho, C.T)     # (n_i, n_f)
            Ainv_D = scipy.linalg.cho_solve(cho, numpy.diag(sqrt_wi))
        else:
            logdet_A = 0.0
            Ainv_Ct = numpy.zeros((0, n_f))
            Ainv_D = numpy.zeros((0, 0))

        Pff = C.dot(Ainv_Ct)                               # (n_f, n_f)
        Pff = 0.5 * (Pff + Pff.T)
        Pfi = C.dot(Ainv_D)                                # (n_f, n_i)
        Pii = sqrt_wi[:, None] * Ainv_D                    # (n_i, n_i)
        Pii = 0.5 * (Pii + Pii.T)

        self.Pff = Pff
        self.Pfi = Pfi
        self.Pii = Pii
        self.Rff = 2.0 * Pff - numpy.eye(n_f)
        self.Rii = 2.0 * Pii - numpy.eye(n_i)
        self.Rfi = 2.0 * Pfi
        self.d = d
        self.delta = _SQRT2 * (d - Pff.dot(d))
        self.eps = -_SQRT2 * Pfi.T.dot(d)

        # log |<0_f|0_i>| through slogdet, so that huge/tiny determinants and
        # exponents never overflow.
        log_pref = (0.5 * n_i * math.log(2.0 * math.pi)
                    - 0.25 * (n_i + n_f) * math.log(math.pi)
                    + 0.25 * (numpy.sum(numpy.log(freq_i)) + numpy.sum(numpy.log(freq_f)))
                    - 0.5 * logdet_A)
        log_exp = -0.5 * float(d.dot(d - Pff.dot(d)))
        self.log_overlap_00 = log_pref + log_exp
        self.overlap_00 = math.exp(self.log_overlap_00) if self.log_overlap_00 > -700.0 else 0.0

    # -- recursion -------------------------------------------------------
    def amplitudes(self, states, init_states=None):
        '''Vector of ``<v_f|w_i>`` for the given occupation vectors.'''
        states = _as_state_array(states, self.n_f, 'states')
        nstate = states.shape[0]
        if init_states is None:
            init = numpy.zeros((nstate, self.n_i), dtype=numpy.int64)
        else:
            init = _as_state_array(init_states, self.n_i, 'init_states')
            if init.shape[0] == 1 and nstate != 1:
                init = numpy.repeat(init, nstate, axis=0)
            if init.shape[0] != nstate:
                raise ValueError(
                    'init_states has %d rows but states has %d; they must be '
                    'paired row by row (or init_states may be a single vector)'
                    % (init.shape[0], nstate))

        memo = {}
        out = numpy.empty(nstate)
        for n in range(nstate):
            out[n] = self._amp(tuple(int(x) for x in states[n]),
                               tuple(int(x) for x in init[n]), memo)
        return out

    def _amp(self, v, w, memo):
        key = (v, w)
        val = memo.get(key)
        if val is not None:
            return val

        delta = self.delta
        Rff = self.Rff
        Rfi = self.Rfi
        Rii = self.Rii
        eps = self.eps

        # Lower one quantum in the first excited final-state mode; when the
        # final state is the vacuum, lower the initial state instead.
        k = -1
        for idx, occ in enumerate(v):
            if occ:
                k = idx
                break
        if k >= 0:
            vk = v[k]
            vp = list(v)
            vp[k] -= 1
            vp = tuple(vp)
            acc = delta[k] * self._amp(vp, w, memo)
            for j, occ in enumerate(vp):
                if occ:
                    r = Rff[k, j]
                    if r != 0.0:
                        vq = list(vp)
                        vq[j] -= 1
                        acc += r * math.sqrt(occ) * self._amp(tuple(vq), w, memo)
            for j, occ in enumerate(w):
                if occ:
                    r = Rfi[k, j]
                    if r != 0.0:
                        wq = list(w)
                        wq[j] -= 1
                        acc += r * math.sqrt(occ) * self._amp(vp, tuple(wq), memo)
            val = acc / math.sqrt(vk)
        else:
            k = -1
            for idx, occ in enumerate(w):
                if occ:
                    k = idx
                    break
            if k < 0:
                val = self.overlap_00
                memo[key] = val
                return val
            wk = w[k]
            wp = list(w)
            wp[k] -= 1
            wp = tuple(wp)
            acc = eps[k] * self._amp(v, wp, memo)
            for j, occ in enumerate(wp):
                if occ:
                    r = Rii[k, j]
                    if r != 0.0:
                        wq = list(wp)
                        wq[j] -= 1
                        acc += r * math.sqrt(occ) * self._amp(v, tuple(wq), memo)
            val = acc / math.sqrt(wk)

        memo[key] = val
        return val


def _as_state_array(states, nmode, name):
    arr = numpy.asarray(states)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != nmode:
        raise ValueError('%s must have shape (nstate, %d), got %s' % (name, nmode, arr.shape))
    out = numpy.rint(arr).astype(numpy.int64)
    if numpy.any(out < 0):
        raise ValueError('%s contains a negative occupation number' % name)
    if numpy.any(numpy.abs(arr - out) > 1e-9):
        raise ValueError('%s must contain integer occupation numbers' % name)
    return out


# ----------------------------------------------------------------------
# 1-D kernels
# ----------------------------------------------------------------------

def _doktorov_1d(omega_i, omega_f, displacement):
    '''Scalar Doktorov parameters for one shared coordinate (J = 1).

    Returns ``(f00, delta, eps, rff, rii, rfi)``.
    '''
    wi = float(omega_i)
    wf = float(omega_f)
    if wi <= FREQ_MIN or wf <= FREQ_MIN:
        raise ValueError('omega_i and omega_f must be positive')
    disp = float(displacement)
    s = wi + wf
    d = math.sqrt(wf) * disp
    pff = wf / s
    pii = wi / s
    pfi = math.sqrt(wi * wf) / s
    f00 = math.sqrt(2.0 * math.sqrt(wi * wf) / s) * math.exp(-0.5 * d * d * (1.0 - pff))
    delta = _SQRT2 * (1.0 - pff) * d
    eps = -_SQRT2 * pfi * d
    return f00, delta, eps, 2.0 * pff - 1.0, 2.0 * pii - 1.0, 2.0 * pfi


def overlap_1d_table(n_f_max, n_i_max, omega_i, omega_f, displacement):
    r'''Table of one-dimensional harmonic overlaps ``<n_f|n_i>``.

    Two oscillators sharing a single coordinate ``Q``, related by
    ``Q_f = Q + displacement`` (i.e. ``J = 1``, ``K = displacement``), with
    angular frequencies ``omega_i`` and ``omega_f``.

    Args:
        n_f_max, n_i_max : int
            Highest quantum number retained for the final/initial oscillator.
        omega_i, omega_f : float
            Angular frequencies in Eh (hbar = 1).
        displacement : float
            Shift of the final-state minimum relative to the initial-state
            minimum, in bohr*sqrt(m_e) (same convention as ``K``).

    Returns:
        ``(n_f_max+1, n_i_max+1)`` ndarray, ``table[n_f, n_i] = <n_f|n_i>``.

    The table is built by the two-index Doktorov recursion (see the module
    docstring), never by independent evaluation of each element.
    '''
    n_f_max = int(n_f_max)
    n_i_max = int(n_i_max)
    if n_f_max < 0 or n_i_max < 0:
        raise ValueError('n_f_max and n_i_max must be non-negative')
    f00, delta, eps, rff, rii, rfi = _doktorov_1d(omega_i, omega_f, displacement)

    tab = numpy.zeros((n_f_max + 1, n_i_max + 1))
    tab[0, 0] = f00
    sq = numpy.sqrt(numpy.arange(max(n_f_max, n_i_max) + 2, dtype=float))

    # Row n_f = 0 by the initial-state recursion (the R_fi term vanishes).
    for ni in range(n_i_max):
        acc = eps * tab[0, ni]
        if ni >= 1:
            acc += rii * sq[ni] * tab[0, ni - 1]
        tab[0, ni + 1] = acc / sq[ni + 1]

    # All remaining rows by the final-state recursion.
    for nf in range(n_f_max):
        for ni in range(n_i_max + 1):
            acc = delta * tab[nf, ni]
            if nf >= 1:
                acc += rff * sq[nf] * tab[nf - 1, ni]
            if ni >= 1:
                acc += rfi * sq[ni] * tab[nf, ni - 1]
            tab[nf + 1, ni] = acc / sq[nf + 1]
    return tab


def overlap_1d(n_f, n_i, omega_i, omega_f, displacement):
    r'''One-dimensional harmonic overlap :math:`\langle n_f | n_i \rangle`.

    General case: the two oscillators may differ in frequency *and* be
    displaced.  See :func:`overlap_1d_table` for the argument convention.

    Returns:
        float

    Note the exact reciprocity relation, which follows from
    ``Q_i = Q_f - displacement`` and the reality of the wavefunctions::

        overlap_1d(n_f, n_i, w_i, w_f,  D) == overlap_1d(n_i, n_f, w_f, w_i, -D)
    '''
    n_f = int(n_f)
    n_i = int(n_i)
    if n_f < 0 or n_i < 0:
        raise ValueError('quantum numbers must be non-negative')
    return float(overlap_1d_table(n_f, n_i, omega_i, omega_f, displacement)[n_f, n_i])


# ----------------------------------------------------------------------
# Multidimensional kernels
# ----------------------------------------------------------------------

def overlap_00(freq_i, freq_f, J, K, cond_tol=COND_TOL):
    r'''Origin overlap :math:`\langle 0_f | 0_i \rangle`, closed form.

    Args:
        freq_i : (n_i,) angular frequencies of the initial state, Eh.
        freq_f : (n_f,) angular frequencies of the final state, Eh.
        J : (n_f, n_i) Duschinsky matrix, ``Q_f = J Q_i + K``.
        K : (n_f,) displacement in bohr*sqrt(m_e).

    Returns:
        float, always positive (the integrand is a positive Gaussian).

    Evaluated through :func:`numpy.linalg.slogdet` so that neither the
    determinant nor the exponent can overflow.  Underflows to exactly ``0.0``
    when the log falls below ``-700``.
    '''
    return _Doktorov(freq_i, freq_f, J, K, cond_tol=cond_tol).overlap_00


#: Alias kept for the name used in the design document.
overlap_0_0 = overlap_00


def multimode_overlaps(freq_i, freq_f, J, K, states, init_states=None,
                       cond_tol=COND_TOL, cache=None):
    r'''Vibrational overlaps :math:`\langle v_f | v_i \rangle` by the Doktorov recursion.

    Handles a completely general (real) Duschinsky matrix ``J``, arbitrary
    frequency changes and arbitrary displacement -- this is *not* a product of
    independent one-dimensional overlaps.  Verified against direct
    Gauss-Hermite quadrature in 1-D, 2-D and 3-D with strongly mixing ``J``.

    Args:
        freq_i : (n_i,) initial-state angular frequencies, Eh.
        freq_f : (n_f,) final-state angular frequencies, Eh.
        J : (n_f, n_i) Duschinsky matrix.
        K : (n_f,) displacement, bohr*sqrt(m_e).
        states : (nstate, n_f) or (n_f,) int array
            Final-state occupation vectors.
        init_states : None, (n_i,) or (nstate, n_i) int array
            Initial-state occupation vectors.  ``None`` means the vibrational
            ground state ``|0_i>`` for every row.  A single vector is
            broadcast; otherwise the rows pair up with ``states``.
        cond_tol : float
            Reject ``A = J^T Omega_f J + Omega_i`` above this condition number.
        cache : :class:`_Doktorov` or None
            Pre-built matrices, to avoid repeating the O(n^3) setup.

    Returns:
        (nstate,) ndarray of overlaps (dimensionless, signed).

    The recursion memoises on ``(v, w)`` so each new state costs O(n_f + n_i)
    given its predecessors.
    '''
    dok = cache if cache is not None else _Doktorov(freq_i, freq_f, J, K, cond_tol=cond_tol)
    return dok.amplitudes(states, init_states)


# ----------------------------------------------------------------------
# State enumeration
# ----------------------------------------------------------------------

def _compositions(total, nparts, qmax):
    '''Ordered tuples of ``nparts`` integers in ``[1, qmax]`` summing to ``total``.'''
    if nparts == 1:
        if 1 <= total <= qmax:
            yield (total,)
        return
    hi = min(qmax, total - (nparts - 1))
    for first in range(1, hi + 1):
        for rest in _compositions(total - first, nparts - 1, qmax):
            yield (first,) + rest


def _count_compositions(total, nparts, qmax):
    '''Number of ordered tuples of ``nparts`` integers in ``[1, qmax]`` summing to ``total``.'''
    if nparts <= 0:
        return 1 if total == 0 else 0
    row = [0] * (total + 1)
    row[0] = 1
    for _ in range(nparts):
        new = [0] * (total + 1)
        for s in range(total + 1):
            if not row[s]:
                continue
            hi = min(qmax, total - s)
            v = row[s]
            for q in range(1, hi + 1):
                new[s + q] += v
        row = new
    return row[total]


def enumerate_states(nmode, max_quanta, max_modes_excited=None,
                     max_quanta_per_mode=None, active_modes=None,
                     max_states=None, hard_limit=2000000):
    '''Class-based enumeration of final-state occupation vectors.

    Following DESIGN.md section 8 (FCclasses-style classes):

    * **class 0** -- the vibrational ground state;
    * **class 1** -- exactly one mode excited.  Always enumerated exhaustively
      over *all* modes, never restricted to ``active_modes``;
    * **class n** (``2 <= n <= max_modes_excited``) -- exactly ``n`` modes
      simultaneously excited, the modes drawn from ``active_modes`` (or from
      all modes when ``active_modes is None``), with total quanta
      ``<= max_quanta`` and each mode ``<= max_quanta_per_mode``.

    Args:
        nmode : int
        max_quanta : int
            Maximum total quanta in a state.
        max_modes_excited : int or None
            Highest class.  ``None`` means ``nmode`` (unbounded).
        max_quanta_per_mode : int or None
            ``None`` means ``max_quanta``.
        active_modes : sequence of int or None
            Mode indices allowed to be *simultaneously* excited (classes >= 2).
        max_states : int or None
            Hard cap on the number of returned states.  Whatever is dropped is
            counted in the returned bookkeeping -- never silently truncated.
        hard_limit : int
            Guard: if ``max_states is None`` and the enumeration would exceed
            this many states, a ``ValueError`` is raised instead of building a
            gigantic array.

    Returns:
        ``(states, info)`` where

        * ``states`` : ``(nstate, nmode)`` ``int16`` array;
        * ``info`` : dict with keys ``n_enumerated`` (the full combinatorial
          count *before* the ``max_states`` cap), ``n_kept``, ``n_skipped``
          (``= n_enumerated - n_kept``), ``truncated`` (bool) and the
          enumeration parameters.

    Ordering is deterministic: states are sorted by
    ``(total quanta, occupation vector lexicographically)``, so two runs give
    byte-identical output.
    '''
    nmode = int(nmode)
    max_quanta = int(max_quanta)
    if nmode < 0:
        raise ValueError('nmode must be non-negative')
    if max_quanta < 0:
        raise ValueError('max_quanta must be non-negative')
    if max_quanta_per_mode is None:
        qmax = max_quanta
    else:
        qmax = min(int(max_quanta_per_mode), max_quanta)
    if max_modes_excited is None:
        nmax = nmode
    else:
        nmax = min(int(max_modes_excited), nmode)
    if nmax < 0:
        raise ValueError('max_modes_excited must be non-negative')

    all_modes = tuple(range(nmode))
    if active_modes is None:
        active = all_modes
    else:
        active = tuple(sorted(set(int(m) for m in active_modes)))
        for m in active:
            if not 0 <= m < nmode:
                raise ValueError('active_modes entry %d out of range [0, %d)' % (m, nmode))

    def _modes_for_class(n):
        return all_modes if n == 1 else active

    # -- analytic total count (cheap, avoids materialising huge lists) ----
    total_count = 1  # class 0
    for t in range(1, max_quanta + 1):
        for n in range(1, min(nmax, t) + 1):
            pool = len(_modes_for_class(n))
            if pool < n:
                continue
            ncomp = _count_compositions(t, n, qmax)
            if ncomp:
                total_count += _binom(pool, n) * ncomp

    if max_states is None and total_count > hard_limit:
        raise ValueError(
            'enumerate_states would produce %d states, exceeding hard_limit=%d.  '
            'Lower max_quanta / max_modes_excited, restrict active_modes, or set '
            'max_states explicitly (the truncation is then reported in the '
            'returned info dict).' % (total_count, hard_limit))

    cap = total_count if max_states is None else min(int(max_states), total_count)

    rows = []
    if cap > 0 and nmode >= 0:
        rows.append(tuple([0] * nmode))
    done = len(rows) >= cap
    for t in range(1, max_quanta + 1):
        if done:
            break
        level = []
        for n in range(1, min(nmax, t) + 1):
            pool = _modes_for_class(n)
            if len(pool) < n:
                continue
            for subset in itertools.combinations(pool, n):
                for comp in _compositions(t, n, qmax):
                    vec = [0] * nmode
                    for m, q in zip(subset, comp):
                        vec[m] = q
                    level.append(tuple(vec))
        level.sort()
        for vec in level:
            rows.append(vec)
            if len(rows) >= cap:
                done = True
                break

    if rows:
        states = numpy.array(rows, dtype=numpy.int16).reshape(len(rows), nmode)
    else:
        states = numpy.zeros((0, nmode), dtype=numpy.int16)

    info = {
        'n_enumerated': int(total_count),
        'n_kept': int(states.shape[0]),
        'n_skipped': int(total_count - states.shape[0]),
        'truncated': bool(total_count > states.shape[0]),
        'max_quanta': max_quanta,
        'max_modes_excited': nmax,
        'max_quanta_per_mode': qmax,
        'max_states': max_states,
        'n_active_modes': len(active),
    }
    return states, info


def _binom(n, k):
    if k < 0 or k > n:
        return 0
    out = 1
    for j in range(k):
        out = out * (n - j) // (j + 1)
    return out


# ----------------------------------------------------------------------
# Result container and driver
# ----------------------------------------------------------------------

class FranckCondonResult(lib.StreamObject):
    '''Container for a harmonic Franck-Condon calculation (DESIGN.md section 4).

    Attributes:
        duschinsky : :class:`pyscf.vibronic.duschinsky.Duschinsky` or None
            ``None`` when the result was produced directly from raw arrays.
        e_00 : float
            Zero-point-corrected origin, ``e_adiabatic + zpe_f - zpe_i`` (Eh).
        e_adiabatic : float
            Bottom-of-well electronic energy difference (Eh).
        states : (nstate, n_f) int16
            Final-state occupation vectors.
        init_states : (nstate, n_i) int16
            Initial-state occupation vectors (all zero at T = 0).
        overlaps : (nstate,) float
            Signed ``<v_f|v_i>``.
        fcf : (nstate,) float
            ``overlaps**2``.
        energies : (nstate,) float
            Signed transition energy ``Delta E`` in Eh,
            ``e_00 + sum_k omega_f,k v_f,k - sum_j omega_i,j v_i,j``.
        populations : (nstate,) float
            Boltzmann population of the initial vibrational state.
        sum_rule : float
            ``sum_v population * |<v_f|v_i>|^2`` over the *enumerated* space,
            computed before intensity pruning.  It converges from below to
            :attr:`sum_rule_target`, which is ``1/|det J|`` and equals 1 only
            when ``J`` is orthogonal.  See :attr:`sum_rule_target`.
        truncation : dict
            Enumeration/pruning bookkeeping.
        freq_i, freq_f : (n_i,), (n_f,) float
            Angular frequencies in Eh.
        temperature : float
            Kelvin.
    '''

    def __init__(self, freq_i, freq_f, states, init_states, overlaps, energies,
                 populations, e_00, e_adiabatic, sum_rule, truncation,
                 duschinsky=None, temperature=0.0, verbose=None, stdout=None):
        self.freq_i = numpy.asarray(freq_i, dtype=float)
        self.freq_f = numpy.asarray(freq_f, dtype=float)
        self.states = numpy.asarray(states, dtype=numpy.int16)
        self.init_states = numpy.asarray(init_states, dtype=numpy.int16)
        self.overlaps = numpy.asarray(overlaps, dtype=float)
        self.fcf = self.overlaps ** 2
        self.energies = numpy.asarray(energies, dtype=float)
        self.populations = numpy.asarray(populations, dtype=float)
        self.e_00 = float(e_00)
        self.e_adiabatic = float(e_adiabatic)
        self.sum_rule = float(sum_rule)
        self.truncation = dict(truncation)
        self.duschinsky = duschinsky
        self.temperature = float(temperature)
        self.verbose = logger.NOTE if verbose is None else verbose
        self.stdout = sys.stdout if stdout is None else stdout

    @property
    def zpe_i(self):
        return 0.5 * float(numpy.sum(self.freq_i))

    @property
    def zpe_f(self):
        return 0.5 * float(numpy.sum(self.freq_f))

    @property
    def nstate(self):
        return self.states.shape[0]

    def stick_spectrum(self, kind='absorption', **kwargs):
        '''Build a :class:`pyscf.vibronic.spectrum.StickSpectrum`.

        All keyword arguments are forwarded to
        :func:`pyscf.vibronic.spectrum.stick_spectrum`.
        '''
        from pyscf.vibronic import spectrum
        return spectrum.stick_spectrum(self, kind=kind, **kwargs)

    @property
    def sum_rule_target(self):
        r'''The exact value the closure rule converges to, ``1/|det J|``.

        The final-state eigenfunctions are complete in :math:`Q_f` space, so
        :math:`\sum_v \psi_f^v(Q_f)\psi_f^v(Q_f') = \delta(Q_f - Q_f')`.
        Substituting :math:`Q_f = J Q_i + K` and integrating over :math:`Q_i`,

        .. math::

            \sum_{v_f} |\langle v_f | 0_i \rangle|^2
              = \frac{1}{|\det J|} \int \mathrm{d}Q_i\, |\psi_i(Q_i)|^2
              = \frac{1}{|\det J|} .

        So the familiar statement "the Franck-Condon factors sum to one" holds
        **only when** :math:`J` is orthogonal, i.e. when both electronic states
        span the same vibrational subspace.  Two different equilibrium
        geometries have slightly different rotational subspaces, so in practice
        :math:`|\det J|` deviates from 1 by ~1e-3 and the sum rule converges to
        a correspondingly shifted value.  Comparing the computed sum against 1
        rather than against this target would misreport that shift as an
        enumeration error.

        Returns 1.0 when no Duschinsky object is attached or ``J`` is not
        square, since ``det J`` is then undefined.
        '''
        dus = self.duschinsky
        if dus is None:
            return 1.0
        j = numpy.asarray(getattr(dus, 'J', None))
        if j.ndim != 2 or j.shape[0] != j.shape[1] or j.size == 0:
            return 1.0
        det = abs(numpy.linalg.det(j))
        if det <= 0:
            return 1.0
        return 1.0 / det

    @property
    def sum_rule_deficit(self):
        '''``sum_rule_target - sum_rule``: the intensity missed by the
        enumeration.  Positive for an incomplete enumeration, ~0 when converged.
        '''
        return self.sum_rule_target - self.sum_rule

    def summary(self):
        '''Multi-line human-readable summary reporting the sum rule honestly.'''
        tr = self.truncation
        target = self.sum_rule_target
        lines = []
        lines.append('FranckCondonResult: %d state(s), n_i = %d, n_f = %d, T = %.3f K'
                     % (self.nstate, self.freq_i.size, self.freq_f.size, self.temperature))
        lines.append('  E_adiabatic = %.10f Eh   ZPE_i = %.10f Eh   ZPE_f = %.10f Eh'
                     % (self.e_adiabatic, self.zpe_i, self.zpe_f))
        lines.append('  E_00        = %.10f Eh = %.2f cm^-1 = %.4f eV'
                     % (self.e_00, units.au2wavenumber(self.e_00), units.au2ev(self.e_00)))
        lines.append('  sum rule    = %.8f  (target 1/|det J| = %.8f, deficit %.3e)'
                     % (self.sum_rule, target, target - self.sum_rule))
        if self.sum_rule < 0.9 * target:
            lines.append('  *** WARNING: the enumerated space captures less than 90% of the '
                         'intensity; increase max_quanta / max_modes_excited. ***')
        lines.append('  enumeration : n_enumerated = %s, n_kept = %s, n_skipped = %s, '
                     'truncated = %s'
                     % (tr.get('n_enumerated'), tr.get('n_kept'), tr.get('n_skipped'),
                        tr.get('truncated')))
        lines.append('  pruning     : intensity_threshold = %s, n_pruned = %s, n_stored = %d'
                     % (tr.get('intensity_threshold'), tr.get('n_pruned'), self.nstate))
        if tr.get('truncated'):
            lines.append('  *** WARNING: %s enumerated state(s) were dropped by max_states. ***'
                         % tr.get('n_skipped'))
        return '\n'.join(lines)

    def __repr__(self):
        return ('<FranckCondonResult nstate=%d E_00=%.6f Eh sum_rule=%.6f truncated=%s>'
                % (self.nstate, self.e_00, self.sum_rule, self.truncation.get('truncated')))

    def __str__(self):
        return self.summary()


def _boltzmann_populations(freq_i, init_states, temperature):
    '''Normalised Boltzmann weights of the listed initial vibrational states.

    The partition function is the *exact* harmonic one (product over modes),
    not the sum over the truncated list, so ``populations`` never sums above 1
    and the deficit is the population missed by the truncation.
    '''
    freq_i = numpy.asarray(freq_i, dtype=float)
    if temperature <= 0.0:
        pops = numpy.zeros(init_states.shape[0])
        ground = numpy.all(init_states == 0, axis=1)
        pops[ground] = 1.0
        return pops
    kt = units.BOLTZMANN_AU * temperature
    x = numpy.exp(-freq_i / kt)
    # Z = prod_k 1/(1 - x_k)   (energies measured from the zero-point level)
    if numpy.any(x >= 1.0):
        raise ValueError('temperature too high for the harmonic partition function')
    logZ = -numpy.sum(numpy.log1p(-x))
    e_rel = init_states.dot(freq_i)
    return numpy.exp(-e_rel / kt - logZ)


def franck_condon_factors(freq_i, freq_f, J, K, e_adiabatic=0.0, max_quanta=4,
                          max_modes_excited=None, max_quanta_per_mode=None,
                          active_modes=None, active_threshold=None,
                          j_offdiag_threshold=0.1, max_states=None,
                          intensity_threshold=0.0, temperature=0.0,
                          max_quanta_init=None, population_threshold=1e-8,
                          duschinsky=None, cond_tol=COND_TOL,
                          hard_limit=2000000, verbose=None, stdout=None):
    r'''Array-level driver: harmonic Franck-Condon factors for ``|0_i> -> |v_f>``.

    Args:
        freq_i : (n_i,) initial-state angular frequencies, Eh.
        freq_f : (n_f,) final-state angular frequencies, Eh.
        J : (n_f, n_i) Duschinsky matrix, ``Q_f = J Q_i + K``.
        K : (n_f,) displacement, bohr*sqrt(m_e).
        e_adiabatic : float
            ``E_elec_f - E_elec_i`` in Eh (bottom of well to bottom of well).
        max_quanta : int
            Maximum total quanta in the final state.
        max_modes_excited, max_quanta_per_mode, active_modes, max_states :
            Passed to :func:`enumerate_states`.
        active_threshold : float or None
            When ``active_modes`` is ``None`` and this is given, the active set
            is ``{k : S_k > active_threshold}`` united with the modes having a
            ``J`` off-diagonal element above ``j_offdiag_threshold``.
        intensity_threshold : float
            States with ``population * fcf`` below this are not stored (they
            are still counted in ``sum_rule`` and in ``truncation['n_pruned']``).
        temperature : float
            Kelvin.  ``0`` means only ``|0_i>`` contributes.  For ``T > 0`` the
            initial states are enumerated as well and the general
            ``<v_f|v_i>`` recursion is used.
        max_quanta_init : int or None
            Total quanta retained in the initial state at ``T > 0``.
            Defaults to ``max_quanta``.
        population_threshold : float
            Initial states with a Boltzmann weight below this are dropped.

    Returns:
        :class:`FranckCondonResult`

    Energies follow DESIGN.md section 3::

        e_00 = e_adiabatic + zpe_f - zpe_i,   zpe = 0.5 * sum(freq)
        Delta E(v_i -> v_f) = e_00 + sum_k omega_f,k v_f,k - sum_j omega_i,j v_i,j
    '''
    log = logger.new_logger(_LogHolder(verbose, stdout), verbose)

    freq_i = _check_freq(freq_i, 'freq_i')
    freq_f = _check_freq(freq_f, 'freq_f')
    n_i = freq_i.size
    n_f = freq_f.size
    J = numpy.asarray(J, dtype=float).reshape(n_f, n_i)
    K = numpy.asarray(K, dtype=float).ravel()

    dok = _Doktorov(freq_i, freq_f, J, K, cond_tol=cond_tol)

    if active_modes is None and active_threshold is not None:
        S = huang_rhys(freq_f, K)
        mask = S > float(active_threshold)
        if n_i and n_f:
            off = numpy.abs(J).copy()
            m = min(n_f, n_i)
            off[numpy.arange(m), numpy.arange(m)] = 0.0
            mask = mask | (off.max(axis=1) > float(j_offdiag_threshold))
        active_modes = numpy.where(mask)[0]
        log.debug('active modes selected by Huang-Rhys/J criteria: %s', list(active_modes))

    states, enum_info = enumerate_states(
        n_f, max_quanta, max_modes_excited=max_modes_excited,
        max_quanta_per_mode=max_quanta_per_mode, active_modes=active_modes,
        max_states=max_states, hard_limit=hard_limit)

    if temperature > 0.0:
        if max_quanta_init is None:
            max_quanta_init = max_quanta
        init_pool, init_info = enumerate_states(
            n_i, max_quanta_init, max_modes_excited=max_modes_excited,
            max_quanta_per_mode=max_quanta_per_mode, active_modes=None,
            max_states=max_states, hard_limit=hard_limit)
        pops_pool = _boltzmann_populations(freq_i, init_pool.astype(numpy.int64), temperature)
        keep = pops_pool >= population_threshold
        init_pool = init_pool[keep]
        pops_pool = pops_pool[keep]
        n_init = init_pool.shape[0]
        n_fin = states.shape[0]
        all_states = numpy.repeat(states, n_init, axis=0)
        all_init = numpy.tile(init_pool, (n_fin, 1))
        populations = numpy.tile(pops_pool, n_fin)
        enum_info = dict(enum_info)
        enum_info['n_enumerated'] = int(enum_info['n_enumerated']) * n_init
        enum_info['n_kept'] = int(all_states.shape[0])
        enum_info['n_skipped'] = enum_info['n_enumerated'] - enum_info['n_kept']
        enum_info['truncated'] = bool(enum_info['n_skipped'] > 0)
        enum_info['n_init_states'] = n_init
        enum_info['init_population_captured'] = float(numpy.sum(pops_pool))
        enum_info['init_enumeration'] = init_info
    else:
        all_states = states
        all_init = numpy.zeros((states.shape[0], n_i), dtype=numpy.int16)
        populations = numpy.ones(states.shape[0])

    overlaps = dok.amplitudes(all_states, all_init)
    if not numpy.all(numpy.isfinite(overlaps)):
        raise RuntimeError(
            'The Doktorov recursion produced non-finite overlaps.  This points '
            'to an extreme displacement or frequency change; reduce max_quanta '
            'or check the inputs.')

    fcf = overlaps ** 2
    weighted = populations * fcf
    sum_rule = float(numpy.sum(weighted))

    zpe_i = 0.5 * float(numpy.sum(freq_i))
    zpe_f = 0.5 * float(numpy.sum(freq_f))
    e_00 = float(e_adiabatic) + zpe_f - zpe_i
    energies = (e_00 + all_states.astype(float).dot(freq_f)
                - all_init.astype(float).dot(freq_i))

    n_before = all_states.shape[0]
    if intensity_threshold and intensity_threshold > 0.0:
        keep = weighted >= float(intensity_threshold)
    else:
        keep = numpy.ones(n_before, dtype=bool)
    n_pruned = int(n_before - numpy.count_nonzero(keep))

    truncation = dict(enum_info)
    truncation['intensity_threshold'] = float(intensity_threshold)
    truncation['n_pruned'] = n_pruned
    truncation['n_stored'] = int(numpy.count_nonzero(keep))
    truncation['temperature'] = float(temperature)

    res = FranckCondonResult(
        freq_i, freq_f, all_states[keep], all_init[keep], overlaps[keep],
        energies[keep], populations[keep], e_00, e_adiabatic, sum_rule,
        truncation, duschinsky=duschinsky, temperature=temperature,
        verbose=verbose, stdout=stdout)

    # The closure rule converges to 1/|det J|, not to 1 (see
    # FranckCondonResult.sum_rule_target), so completeness must be judged
    # against that target.
    target = res.sum_rule_target
    if sum_rule < 0.9 * target:
        log.warn('Franck-Condon sum rule is only %.6f of a target %.6f (= 1/|det J|); '
                 'the enumerated space misses %.2f%% of the intensity.  Increase '
                 'max_quanta or max_modes_excited.',
                 sum_rule, target, 100.0 * (1.0 - sum_rule / target))
    if truncation.get('truncated'):
        log.warn('%d enumerated state(s) were dropped by max_states=%s.',
                 truncation.get('n_skipped'), truncation.get('max_states'))
    log.debug('Franck-Condon: %d state(s) stored, sum rule %.8f (target %.8f)',
              res.nstate, sum_rule, target)
    return res


class _LogHolder(object):
    '''Minimal carrier so :func:`pyscf.lib.logger.new_logger` can be used.'''

    def __init__(self, verbose, stdout):
        self.verbose = logger.NOTE if verbose is None else verbose
        self.stdout = sys.stdout if stdout is None else stdout
