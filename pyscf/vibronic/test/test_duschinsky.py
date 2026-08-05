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

'''Tests for pyscf.vibronic.duschinsky.

Most cases are built *synthetically* so that they are fast and exact.  The key
trick (``_model_state`` below) is that a Cartesian Hessian with prescribed
vibrational eigenpairs is

    H = M^{1/2} L diag(lambda) L^T M^{1/2}

because ``M^{-1/2} H M^{-1/2} = L diag(lambda) L^T`` vanishes on the
translation/rotation subspace and has exactly the eigenpairs ``(lambda_k, L_k)``
on the vibrational subspace.  Feeding this Hessian through the *real*
``HarmonicModel`` pipeline therefore returns ``freq = sqrt(lambda)`` and
``modes = L`` exactly (up to the per-column sign convention), which lets every
quantity be compared against a hand-derived value while still exercising the
production code path (rotor classification, translation/rotation projection,
mass weighting, phase fixing).

One genuinely *ab initio* case (RHF/STO-3G water at two displaced geometries) is
included at the end.
'''

import io
import math
import unittest
import numpy

from pyscf import gto, scf, lib
from pyscf.vibronic import alignment, duschinsky, normal_modes, units


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def random_rotation(seed):
    '''A random proper rotation (det = +1).'''
    rs = numpy.random.RandomState(seed)
    q, r = numpy.linalg.qr(rs.randn(3, 3))
    q = q * numpy.sign(numpy.diag(r))
    if numpy.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def random_orthogonal(n, seed):
    '''A random orthogonal ``(n,n)`` matrix, deterministic in ``seed``.'''
    rs = numpy.random.RandomState(seed)
    q, r = numpy.linalg.qr(rs.randn(n, n))
    return q * numpy.sign(numpy.diag(r))


def vib_basis(mass_au, coords):
    '''Orthonormal basis of the vibrational subspace, ``(3N, nvib)``.'''
    return normal_modes.projector_trans_rot(mass_au, coords)[1]


def cartesian_hessian(mass_au, modes, force_const):
    '''``H = M^{1/2} L diag(lambda) L^T M^{1/2}``, the Cartesian Hessian with
    the prescribed vibrational eigenpairs and exactly zero force constant along
    translations and rotations.
    '''
    sqrt_m = numpy.repeat(numpy.sqrt(mass_au), 3)
    hmw = (modes * numpy.asarray(force_const)).dot(modes.T)
    return hmw * sqrt_m[:, None] * sqrt_m[None, :]


def rotate_hessian(hess, rot):
    '''Rotate a ``(3N,3N)`` Cartesian Hessian consistently with ``coords @ rot``.

    ``x'_{a x} = sum_c x_{a c} R_{c x}`` implies
    ``H'_{ax,by} = sum_{cd} R_{cx} H_{ac,bd} R_{dy}``.
    '''
    natm = hess.shape[0] // 3
    h4 = hess.reshape(natm, 3, natm, 3)
    return numpy.einsum('acbd,cx,dy->axby', h4, rot, rot).reshape(3 * natm, 3 * natm)


def water_coords(r1=0.989, r2=0.989, angle=100.0):
    '''(natm,3) water geometry in **bohr**, O first.'''
    a = math.radians(angle)
    ang = numpy.array([[0., 0., 0.],
                       [0., 0., r1],
                       [0., r2 * math.sin(a), r2 * math.cos(a)]])
    return ang / units.BOHR


def nonplanar_coords():
    '''A deliberately low-symmetry, non-planar 4-atom geometry, in bohr.'''
    return numpy.array([[0.10, -0.20, 0.05],
                        [1.90, 0.15, -0.30],
                        [-0.60, 1.75, 0.40],
                        [-0.35, -0.55, -1.85]])


class State(object):
    '''A synthetic electronic state: masses, geometry, prescribed L and freq,
    the resulting Cartesian Hessian, and the ``HarmonicModel`` built from it.
    '''

    def __init__(self, charges, mass_amu, coords, modes, freq, imaginary=(),
                 **kw):
        self.atom_charges = numpy.asarray(charges, dtype=int)
        self.mass_amu = numpy.asarray(mass_amu, dtype=float)
        self.mass = numpy.asarray(units.amu2au(self.mass_amu), dtype=float)
        self.coords = alignment.shift_to_center_of_mass(self.mass, coords)
        self.modes = numpy.asarray(modes, dtype=float)
        self.freq = numpy.asarray(freq, dtype=float)
        fc = self.freq**2 * numpy.where(numpy.isin(numpy.arange(self.freq.size),
                                                   numpy.asarray(imaginary, dtype=int)),
                                        -1., 1.)
        self.force_const = fc
        self.hessian = cartesian_hessian(self.mass, self.modes, fc)
        self.model = normal_modes.HarmonicModel(
            self.atom_charges, self.coords, self.mass_amu, self.hessian,
            verbose=0, **kw)


def make_state(coords, charges, mass_amu, freq, seed, imaginary=(), **kw):
    '''Synthetic state with a random orthogonal mixture of the vibrational
    basis as its mode matrix and the prescribed frequencies (a.u.).
    '''
    mass = numpy.asarray(units.amu2au(numpy.asarray(mass_amu, dtype=float)))
    coords_c = alignment.shift_to_center_of_mass(mass, coords)
    basis = vib_basis(mass, coords_c)
    freq = numpy.asarray(freq, dtype=float)
    assert basis.shape[1] == freq.size, (basis.shape, freq.size)
    modes = basis.dot(random_orthogonal(freq.size, seed))
    return State(charges, mass_amu, coords_c, modes, freq, imaginary=imaginary, **kw)


WATER_CHARGES = [8, 1, 1]
WATER_MASS = [15.9994, 1.008, 1.008]
#: three well-separated frequencies, ~1600 / 3600 / 3900 cm^-1
WATER_FREQ = numpy.array([0.0073, 0.0164, 0.0178])


def recover_sign_pattern(J_ref, J, thresh=None):
    '''Signs ``(s_f, s_i)`` such that ``diag(s_f) J diag(s_i) == J_ref``.

    The columns of ``L_i``/``L_f`` are defined only up to a sign, so a phase
    re-fixing acts as ``J -> D_f J D_i``.  Recovering ``D_f`` and ``D_i``
    row-by-row and then column-by-column is *under-determined* (a row sign and a
    column sign can absorb each other), so the signs are propagated over the
    bipartite graph of significant elements of ``J_ref`` instead: fix
    ``s_f[0] = +1``, then alternate

        ``s_i[j] = sign(J_ref[k,j] J[k,j]) s_f[k]``,
        ``s_f[k] = sign(J_ref[k,j] J[k,j]) s_i[j]``

    over every element above ``thresh``.  The result is unique up to the global
    gauge ``(s_f, s_i) -> (-s_f, -s_i)``, which leaves ``diag(s_f) J diag(s_i)``
    unchanged.  If ``J`` is not of the form ``D_f J_ref D_i`` the reconstruction
    will not match ``J_ref`` and the caller's assertion fails -- which is the
    point.
    '''
    J_ref = numpy.asarray(J_ref, dtype=float)
    J = numpy.asarray(J, dtype=float)
    nf, ni = J_ref.shape
    if thresh is None:
        thresh = 1e-3 * abs(J_ref).max()
    s_f = [None] * nf
    s_i = [None] * ni
    for start in range(nf):
        if s_f[start] is not None:
            continue
        s_f[start] = 1.
        stack = [('f', start)]
        while stack:
            side, idx = stack.pop()
            if side == 'f':
                for j in range(ni):
                    if abs(J_ref[idx, j]) > thresh and s_i[j] is None:
                        s_i[j] = math.copysign(1., J_ref[idx, j] * J[idx, j]) * s_f[idx]
                        stack.append(('i', j))
            else:
                for k in range(nf):
                    if abs(J_ref[k, idx]) > thresh and s_f[k] is None:
                        s_f[k] = math.copysign(1., J_ref[k, idx] * J[k, idx]) * s_i[idx]
                        stack.append(('f', k))
    s_f = numpy.array([1. if x is None else x for x in s_f])
    s_i = numpy.array([1. if x is None else x for x in s_i])
    return s_f, s_i


def sign_equivalence_errors(J_ref, K_ref, J, K):
    '''``(err_J, err_K)`` for the claim ``J = D_f J_ref D_i``, ``K = D_f K_ref``.

    ``err_J`` is gauge-independent (the global gauge cancels in the product).
    ``err_K`` is minimised over the one remaining global sign, which costs a
    single bit of information; the *absolute* sign of ``K`` is pinned separately
    and phase-freely by ``L_f K == M^{1/2}(x_i0 - x_f0)``, asserted alongside
    every use of this helper.
    '''
    s_f, s_i = recover_sign_pattern(J_ref, J)
    err_J = float(abs(s_f[:, None] * J * s_i - J_ref).max())
    err_K = float(min(abs(s_f * K - K_ref).max(), abs(-s_f * K - K_ref).max()))
    return err_J, err_K


def assert_displacement_sign(test, dusch, places=12, projected=False):
    '''``L_f K == M^{1/2}(x_i0 - x_f0)``: the phase-free sign anchor.

    Exact only when the mass-weighted geometry change lies entirely in the
    final-state vibrational subspace.  With ``projected=True`` the right-hand
    side is projected with ``L_f L_f^T`` first, which holds identically and is
    the right form when the alignment is deliberately imperfect.
    '''
    sqrt_m = numpy.repeat(numpy.sqrt(dusch.mass), 3)
    d_expect = sqrt_m * (dusch.coords_i - dusch.coords_f).ravel()
    if projected:
        d_expect = dusch.modes_f.dot(dusch.modes_f.T.dot(d_expect))
    test.assertAlmostEqual(abs(dusch.modes_f.dot(dusch.K) - d_expect).max(), 0., places)


def greedy_assignment(w):
    '''Greedy global-descending assignment of the benefit matrix ``w``.

    Repeatedly take the largest remaining element whose row and column are both
    free.  Returns ``(pairs, total)``.
    '''
    w = numpy.array(w, dtype=float)
    nf, ni = w.shape
    free_f = set(range(nf))
    free_i = set(range(ni))
    pairs = []
    total = 0.
    while free_f and free_i:
        best = None
        for k in sorted(free_f):
            for j in sorted(free_i):
                if best is None or w[k, j] > w[best]:
                    best = (k, j)
        pairs.append(best)
        total += w[best]
        free_f.discard(best[0])
        free_i.discard(best[1])
    return pairs, total


# ---------------------------------------------------------------------------
# the ab initio reference case, computed once
# ---------------------------------------------------------------------------

_REAL = {}


def setUpModule():
    def build(coords_bohr):
        mol = gto.Mole()
        mol.atom = [('O', coords_bohr[0]), ('H', coords_bohr[1]), ('H', coords_bohr[2])]
        mol.unit = 'Bohr'
        mol.basis = 'sto-3g'
        mol.verbose = 0
        mol.build()
        mf = scf.RHF(mol).run()
        hess = mf.Hessian().kernel()
        return normal_modes.HarmonicModel.from_mole(mol, hess, energy=mf.e_tot, verbose=0)

    _REAL['i'] = build(water_coords(0.989, 0.989, 100.0))
    _REAL['f'] = build(water_coords(1.060, 0.950, 112.0))


def tearDownModule():
    _REAL.clear()


class KnownValues(unittest.TestCase):

    # -- identity and pure displacement ----------------------------------

    def test_identity_limit(self):
        '''Identical states -> J = 1, K = 0.'''
        st = make_state(water_coords(), WATER_CHARGES, WATER_MASS, WATER_FREQ, seed=11)
        d = duschinsky.duschinsky_transform(st.model, st.model, verbose=0)
        nv = st.model.nvib
        self.assertEqual(d.J.shape, (nv, nv))
        self.assertAlmostEqual(abs(d.J - numpy.eye(nv)).max(), 0., 12)
        self.assertAlmostEqual(abs(d.K).max(), 0., 12)
        self.assertAlmostEqual(abs(d.huang_rhys).max(), 0., 12)
        self.assertAlmostEqual(d.total_reorganization_energy, 0., 12)
        self.assertAlmostEqual(d.diagnostics['orthogonality_error'], 0., 12)
        self.assertAlmostEqual(abs(d.diagnostics['det_J'] - 1.), 0., 12)

    def test_identity_limit_from_arrays(self):
        '''Same, through the low-level entry point with a hand-built L.'''
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x = alignment.shift_to_center_of_mass(mass, water_coords())
        L = vib_basis(mass, x).dot(random_orthogonal(3, 5))
        d = duschinsky.duschinsky_from_arrays(L, L, mass, x, x, WATER_FREQ, WATER_FREQ,
                                              verbose=0)
        self.assertAlmostEqual(abs(d.J - numpy.eye(3)).max(), 0., 14)
        # K is a difference of mass-weighted coordinates of magnitude
        # |M^{1/2} x| ~ 1e2 bohr sqrt(m_e), so its floor is ~1e2 * eps ~ 2e-14,
        # not 1e-16.
        self.assertAlmostEqual(abs(d.K).max(), 0., 12)

    def test_pure_displacement_sign(self):
        '''Same Hessian, geometry translated along one normal mode.

        With ``x_i0 = x_f0 + s * M^{-1/2} L_k`` (``s > 0``) the mass-weighted
        displacement is ``d = M^{1/2}(x_i0 - x_f0) = s L_k`` exactly, so
        ``K = L^T d = s e_k``.  A displacement along a vibrational mode is
        exactly orthogonal to both the translations and the rotations (the
        vibrational basis is built as the orthogonal complement of those), so the
        centre of mass and the Eckart frame are unchanged to machine precision
        and the result is exact with *and* without the alignment step.

        **This is the sign test.** If ``K`` were defined as
        ``L_f^T M^{1/2}(x_f0 - x_i0)`` every ``K[k]`` below would come out
        negative and every assertion on the sign would fail.
        '''
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x_f = alignment.shift_to_center_of_mass(mass, water_coords())
        L = vib_basis(mass, x_f).dot(random_orthogonal(3, 7))
        inv_sqrt_m = numpy.repeat(mass**-0.5, 3)

        for k in range(3):
            for s in (0.35, -0.35):
                shift = (s * inv_sqrt_m * L[:, k]).reshape(-1, 3)
                x_i = x_f + shift
                # the centre of mass really is unchanged
                self.assertAlmostEqual(
                    abs(alignment.center_of_mass(mass, x_i)
                        - alignment.center_of_mass(mass, x_f)).max(), 0., 13)
                expect = numpy.zeros(3)
                expect[k] = s
                for align in (False, True):
                    d = duschinsky.duschinsky_from_arrays(
                        L, L, mass, x_i, x_f, WATER_FREQ, WATER_FREQ,
                        align=align, verbose=0)
                    self.assertAlmostEqual(abs(d.J - numpy.eye(3)).max(), 0., 12)
                    self.assertAlmostEqual(abs(d.K - expect).max(), 0., 12)
                    self.assertEqual(numpy.sign(d.K[k]), numpy.sign(s))
                    # Huang-Rhys from the hand-derived K
                    s_expect = 0.5 * WATER_FREQ[k] * s**2
                    self.assertAlmostEqual(d.huang_rhys[k], s_expect, 14)
                    self.assertAlmostEqual(d.reorganization_energy[k],
                                           s_expect * WATER_FREQ[k], 16)

    def test_displacement_reconstruction_sign_is_phase_free(self):
        '''``L_f K == M^{1/2}(x_i0 - x_f0)``, a *phase-independent* sign check.

        Every element of ``K`` flips sign with the arbitrary phase of the
        corresponding column of ``L_f``, but ``L_f K`` does not.  Asserting
        ``L_f K = +d`` (not ``-d``) therefore pins the sign convention down
        without reference to any phase choice.
        '''
        st_i = make_state(water_coords(0.989, 0.989, 100.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ, seed=3)
        st_f = make_state(water_coords(1.04, 0.96, 108.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ * 0.9, seed=4)
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        sqrt_m = numpy.repeat(numpy.sqrt(d.mass), 3)
        d_expect = sqrt_m * (d.coords_i - d.coords_f).ravel()
        self.assertAlmostEqual(abs(d.displacement - d_expect).max(), 0., 14)
        self.assertAlmostEqual(abs(d.modes_f.dot(d.K) - d_expect).max(), 0., 12)
        # ... and it is definitely not -d
        self.assertGreater(abs(d.modes_f.dot(d.K) + d_expect).max(), 1.0)

    # -- the relation itself ---------------------------------------------

    def test_apply_round_trip(self):
        '''``apply(Q_i)`` reproduces ``L_f^T (L_i Q_i + d)``.'''
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x_i = alignment.shift_to_center_of_mass(mass, water_coords(0.989, 0.989, 100.))
        x_f = alignment.shift_to_center_of_mass(mass, water_coords(1.04, 0.96, 108.))
        L_i = vib_basis(mass, x_i).dot(random_orthogonal(3, 21))
        L_f = vib_basis(mass, x_f).dot(random_orthogonal(3, 22))
        # align=False so that the test needs nothing from the implementation
        d = duschinsky.duschinsky_from_arrays(L_i, L_f, mass, x_i, x_f,
                                              WATER_FREQ, WATER_FREQ, align=False,
                                              verbose=0)
        sqrt_m = numpy.repeat(numpy.sqrt(mass), 3)
        disp = sqrt_m * (x_i - x_f).ravel()
        rs = numpy.random.RandomState(23)
        for _ in range(5):
            Q_i = rs.randn(3)
            ref = L_f.T.dot(L_i.dot(Q_i) + disp)
            self.assertAlmostEqual(abs(d.apply(Q_i) - ref).max(), 0., 12)
        # batched input
        Q = rs.randn(4, 3)
        ref = numpy.array([L_f.T.dot(L_i.dot(q) + disp) for q in Q])
        self.assertAlmostEqual(abs(d.apply(Q) - ref).max(), 0., 12)
        self.assertRaises(ValueError, d.apply, numpy.zeros(2))

        # and with the alignment on, using the stored (rotated) arrays
        da = duschinsky.duschinsky_from_arrays(L_i, L_f, mass, x_i, x_f,
                                               WATER_FREQ, WATER_FREQ, verbose=0)
        for _ in range(5):
            Q_i = rs.randn(3)
            ref = da.modes_f.T.dot(da.modes_i.dot(Q_i) + da.displacement)
            self.assertAlmostEqual(abs(da.apply(Q_i) - ref).max(), 0., 12)

    def test_orthogonality_same_geometry(self):
        '''Two states at the *same* geometry share the TR subspace -> J is
        exactly orthogonal.'''
        coords = water_coords()
        st_i = make_state(coords, WATER_CHARGES, WATER_MASS, WATER_FREQ, seed=31)
        st_f = make_state(coords, WATER_CHARGES, WATER_MASS, WATER_FREQ * 0.8, seed=32)
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        self.assertAlmostEqual(d.diagnostics['orthogonality_error'], 0., 10)
        self.assertAlmostEqual(d.diagnostics['row_orthogonality_error'], 0., 10)
        self.assertAlmostEqual(abs(abs(d.diagnostics['det_J']) - 1.), 0., 10)
        self.assertAlmostEqual(abs(d.K).max(), 0., 10)
        sv = d.diagnostics['subspace_overlap']
        self.assertAlmostEqual(abs(sv - 1.).max(), 0., 10)

    # -- invariances ------------------------------------------------------

    def test_rotation_invariance_arrays(self):
        '''Rotating + translating the final state (geometry *and* modes) leaves
        J, K and S unchanged -- exactly, because no re-diagonalisation and hence
        no phase re-fixing is involved.
        '''
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x_i = alignment.shift_to_center_of_mass(mass, water_coords(0.989, 0.989, 100.))
        x_f = alignment.shift_to_center_of_mass(mass, water_coords(1.04, 0.96, 108.))
        L_i = vib_basis(mass, x_i).dot(random_orthogonal(3, 41))
        L_f = vib_basis(mass, x_f).dot(random_orthogonal(3, 42))
        ref = duschinsky.duschinsky_from_arrays(L_i, L_f, mass, x_i, x_f,
                                                WATER_FREQ, WATER_FREQ, verbose=0)
        for seed in (1, 2, 3):
            rot = random_rotation(seed)
            trans = numpy.array([0.7, -1.3, 2.9])
            got = duschinsky.duschinsky_from_arrays(
                L_i, duschinsky.rotate_modes(L_f, rot), mass, x_i,
                x_f.dot(rot) + trans, WATER_FREQ, WATER_FREQ, verbose=0)
            self.assertAlmostEqual(abs(got.J - ref.J).max(), 0., 12)
            self.assertAlmostEqual(abs(got.K - ref.K).max(), 0., 12)
            self.assertAlmostEqual(abs(got.huang_rhys - ref.huang_rhys).max(), 0., 14)
            # the same, for the *initial* state
            got_i = duschinsky.duschinsky_from_arrays(
                duschinsky.rotate_modes(L_i, rot), L_f, mass,
                x_i.dot(rot) + trans, x_f, WATER_FREQ, WATER_FREQ, verbose=0)
            self.assertAlmostEqual(abs(got_i.J - ref.J).max(), 0., 12)
            self.assertAlmostEqual(abs(got_i.K - ref.K).max(), 0., 12)

    def test_rotation_invariance_rediagonalised(self):
        '''The same, but recomputing the modes from a correctly rotated Hessian.

        ``H'_{ax,by} = R_{cx} H_{ac,bd} R_{dy}``.  Re-diagonalising in the
        rotated frame re-fixes the per-column phase of ``L_f`` (the phase
        convention picks the largest component, which a rotation can move), so
        the invariant statement is ``J' = D_f J D_i`` and ``K' = D_f K`` with the
        **same** ``D_f`` -- and every phase-free quantity is unchanged outright.
        '''
        st_i = make_state(water_coords(0.989, 0.989, 100.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ, seed=51)
        st_f = make_state(water_coords(1.04, 0.96, 108.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ * 0.85, seed=52)
        ref = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)

        for seed, which in ((4, 'f'), (5, 'f'), (6, 'i'), (7, 'i')):
            rot = random_rotation(seed)
            trans = numpy.array([-2.1, 0.4, 1.7])
            st = st_f if which == 'f' else st_i
            model_rot = normal_modes.HarmonicModel(
                st.atom_charges, st.coords.dot(rot) + trans, st.mass_amu,
                rotate_hessian(st.hessian, rot), verbose=0)
            self.assertAlmostEqual(
                abs(model_rot.freq - st.model.freq).max(), 0., 12)
            if which == 'f':
                got = duschinsky.duschinsky_transform(st_i.model, model_rot, verbose=0)
            else:
                got = duschinsky.duschinsky_transform(model_rot, st_f.model, verbose=0)

            err_J, err_K = sign_equivalence_errors(ref.J, ref.K, got.J, got.K)
            self.assertAlmostEqual(err_J, 0., 9)
            self.assertAlmostEqual(err_K, 0., 9)
            assert_displacement_sign(self, got)
            assert_displacement_sign(self, ref)
            self.assertAlmostEqual(abs(abs(got.K) - abs(ref.K)).max(), 0., 9)
            self.assertAlmostEqual(abs(got.huang_rhys - ref.huang_rhys).max(), 0., 12)
            self.assertAlmostEqual(
                abs(got.reorganization_energy - ref.reorganization_energy).max(), 0., 14)
            self.assertAlmostEqual(
                abs(numpy.sort(got.diagnostics['subspace_overlap'])
                    - numpy.sort(ref.diagnostics['subspace_overlap'])).max(), 0., 10)

    def test_atom_reordering_invariance(self):
        '''Consistently permuting the atoms in *both* states changes nothing
        physical.

        The modes are ordered by frequency, so the *mode* ordering is untouched;
        the rows of ``L`` are permuted, which leaves ``J = L_f^T L_i`` and
        ``K = L_f^T d`` algebraically identical.  The only thing that can change
        is the per-column sign convention (the phase pivot is the largest
        component, whose *index* moves under the permutation), so what is
        asserted is ``abs(J)``, ``abs(K)``, ``huang_rhys``, the singular values,
        and the full sign pattern ``J' = D_f J D_i`` / ``K' = D_f K``.

        A low-symmetry random model Hessian is used so there are no degeneracies
        that could legitimately re-mix.
        '''
        coords_i = nonplanar_coords()
        coords_f = coords_i + numpy.array([[0.05, -0.03, 0.02],
                                           [-0.09, 0.06, 0.11],
                                           [0.04, 0.12, -0.07],
                                           [0.03, -0.10, -0.05]])
        charges = numpy.array([6, 1, 8, 7])
        mass_amu = numpy.array([12.011, 1.008, 15.9994, 14.007])
        freq_i = numpy.array([0.004, 0.007, 0.010, 0.013, 0.016, 0.019])
        freq_f = freq_i * numpy.array([0.9, 1.1, 0.95, 1.05, 0.85, 1.15])
        st_i = make_state(coords_i, charges, mass_amu, freq_i, seed=61)
        st_f = make_state(coords_f, charges, mass_amu, freq_f, seed=62)
        ref = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)

        for perm in ([2, 0, 3, 1], [3, 2, 1, 0]):
            perm = numpy.asarray(perm)
            idx = (3 * perm[:, None] + numpy.arange(3)).ravel()
            states = []
            for st in (st_i, st_f):
                states.append(normal_modes.HarmonicModel(
                    st.atom_charges[perm], st.coords[perm], st.mass_amu[perm],
                    st.hessian[numpy.ix_(idx, idx)], verbose=0))
            got = duschinsky.duschinsky_transform(states[0], states[1], verbose=0)
            self.assertAlmostEqual(abs(got.freq_i - ref.freq_i).max(), 0., 12)
            self.assertAlmostEqual(abs(abs(got.J) - abs(ref.J)).max(), 0., 9)
            self.assertAlmostEqual(abs(abs(got.K) - abs(ref.K)).max(), 0., 9)
            self.assertAlmostEqual(abs(got.huang_rhys - ref.huang_rhys).max(), 0., 12)
            err_J, err_K = sign_equivalence_errors(ref.J, ref.K, got.J, got.K)
            self.assertAlmostEqual(err_J, 0., 9)
            self.assertAlmostEqual(err_K, 0., 9)
            assert_displacement_sign(self, got)

    def test_normal_mode_phase_invariance(self):
        '''Flipping the sign of arbitrary columns of L_i / L_f flips rows and
        columns of J and entries of K, but changes nothing physical.'''
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x_i = alignment.shift_to_center_of_mass(mass, water_coords(0.989, 0.989, 100.))
        x_f = alignment.shift_to_center_of_mass(mass, water_coords(1.04, 0.96, 108.))
        L_i = vib_basis(mass, x_i).dot(random_orthogonal(3, 71))
        L_f = vib_basis(mass, x_f).dot(random_orthogonal(3, 72))
        ref = duschinsky.duschinsky_from_arrays(L_i, L_f, mass, x_i, x_f,
                                                WATER_FREQ, WATER_FREQ, verbose=0)
        for flip_i, flip_f in (((1,), ()), ((), (0, 2)), ((0, 2), (1, 2)), ((0, 1, 2),) * 2):
            s_i = numpy.ones(3)
            s_i[list(flip_i)] = -1.
            s_f = numpy.ones(3)
            s_f[list(flip_f)] = -1.
            got = duschinsky.duschinsky_from_arrays(L_i * s_i, L_f * s_f, mass, x_i, x_f,
                                                    WATER_FREQ, WATER_FREQ, verbose=0)
            # J and K change exactly as predicted
            self.assertAlmostEqual(abs(got.J - s_f[:, None] * ref.J * s_i).max(), 0., 14)
            self.assertAlmostEqual(abs(got.K - s_f * ref.K).max(), 0., 14)
            # everything physical is invariant
            self.assertAlmostEqual(abs(got.huang_rhys - ref.huang_rhys).max(), 0., 14)
            self.assertAlmostEqual(abs(abs(got.J) - abs(ref.J)).max(), 0., 14)
            self.assertAlmostEqual(
                abs(got.diagnostics['subspace_overlap']
                    - ref.diagnostics['subspace_overlap']).max(), 0., 14)
            self.assertAlmostEqual(got.total_reorganization_energy,
                                   ref.total_reorganization_energy, 14)
            self.assertAlmostEqual(got.diagnostics['excluded_mode_norm'],
                                   ref.diagnostics['excluded_mode_norm'], 14)

    # -- analytic 1-D model ----------------------------------------------

    def test_huang_rhys_1d_diatomic(self):
        '''Displaced diatomic with a known force constant.

        For ``E = k(r-r0)^2/2`` the single vibration has ``omega = sqrt(k/mu)``
        with ``mu = m_H m_F/(m_H+m_F)``.  A bond-length change ``dr`` between the
        two states gives, with both structures at their centre of mass,

            ``|K| = sqrt(mu)*|dr|``,
            ``S   = omega K^2/2 = sqrt(k*mu) dr^2 / 2``,
            ``lambda = S*omega = k dr^2 / 2``

        the last of which is just the final-state harmonic energy at the
        initial-state geometry -- an independent route to the same number.
        '''
        m_amu = numpy.array([1.008, 18.998403])
        mass = numpy.asarray(units.amu2au(m_amu))
        mu = mass[0] * mass[1] / mass.sum()
        k = 0.62                      # Eh/bohr^2, close to real HF
        r_i, r_f = 1.7328, 1.8500     # bohr

        def diatomic(r):
            hess = numpy.zeros((6, 6))
            hess[2, 2] = hess[5, 5] = k
            hess[2, 5] = hess[5, 2] = -k
            coords = numpy.array([[0., 0., 0.], [0., 0., r]])
            return normal_modes.HarmonicModel([1, 9], coords, m_amu, hess, verbose=0)

        mi, mf = diatomic(r_i), diatomic(r_f)
        self.assertEqual(mi.rotor_type, 'LINEAR')
        self.assertEqual(mi.nvib, 1)
        omega = math.sqrt(k / mu)
        self.assertAlmostEqual(mi.freq[0], omega, 12)
        self.assertAlmostEqual(mf.freq[0], omega, 12)

        d = duschinsky.duschinsky_transform(mi, mf, verbose=0)
        dr = r_i - r_f
        self.assertAlmostEqual(abs(d.J[0, 0]), 1., 12)
        self.assertAlmostEqual(abs(d.K[0]), math.sqrt(mu) * abs(dr), 10)
        # sign, phase-free: L_f K must reproduce +M^{1/2}(x_i0 - x_f0)
        sqrt_m = numpy.repeat(numpy.sqrt(mass), 3)
        self.assertAlmostEqual(
            abs(d.modes_f.dot(d.K) - sqrt_m * (d.coords_i - d.coords_f).ravel()).max(),
            0., 12)

        s_expect = 0.5 * math.sqrt(k * mu) * dr**2
        self.assertAlmostEqual(d.huang_rhys[0], s_expect, 12)
        self.assertAlmostEqual(0.5 * omega * d.K[0]**2, s_expect, 12)
        lam_expect = 0.5 * k * dr**2
        self.assertAlmostEqual(d.reorganization_energy[0], lam_expect, 12)
        self.assertAlmostEqual(d.total_reorganization_energy, lam_expect, 12)
        self.assertAlmostEqual(d.diagnostics['total_reorganization_energy_cm'],
                               units.au2wavenumber(lam_expect), 6)
        # nothing leaks out of the (1-dimensional) vibrational subspace
        self.assertLess(d.diagnostics['excluded_mode_norm'], 1e-25)

    # -- degeneracy -------------------------------------------------------

    def test_degenerate_block_remixing(self):
        '''A degenerate final-state pair is reported as a *block*, and the
        block-to-block overlap is invariant under an arbitrary orthogonal
        re-mixing of that block -- while the individual J elements are not.
        '''
        coords_i = nonplanar_coords()
        coords_f = coords_i * 1.01
        charges = numpy.array([6, 1, 8, 7])
        mass_amu = numpy.array([12.011, 1.008, 15.9994, 14.007])
        mass = numpy.asarray(units.amu2au(mass_amu))
        x_i = alignment.shift_to_center_of_mass(mass, coords_i)
        x_f = alignment.shift_to_center_of_mass(mass, coords_f)
        L_i = vib_basis(mass, x_i).dot(random_orthogonal(6, 81))
        L_f = vib_basis(mass, x_f).dot(random_orthogonal(6, 82))
        freq_i = numpy.array([0.004, 0.007, 0.010, 0.013, 0.016, 0.019])
        # modes 2 and 3 of the final state are exactly degenerate
        freq_f = numpy.array([0.0035, 0.0065, 0.0120, 0.0120, 0.0150, 0.0185])

        ref = duschinsky.duschinsky_from_arrays(L_i, L_f, mass, x_i, x_f,
                                                freq_i, freq_f, verbose=0)
        m_ref = duschinsky.match_modes(ref)
        blocks_f = [b for b in m_ref.blocks_f if len(b) > 1]
        self.assertEqual(blocks_f, [[2, 3]])
        self.assertEqual([b for b in m_ref.blocks_i if len(b) > 1], [])

        deg = [m for m in m_ref if m.degenerate]
        self.assertEqual(len(deg), 2)
        for m in deg:
            self.assertIsNone(m.mode_f)                 # no individual claim
            self.assertEqual(m.block_f, m_ref.block_of_f[2])
            self.assertEqual(len(m.block_singular_values), 1)
        # the unambiguous mapping deliberately omits the degenerate entries
        self.assertEqual(len(m_ref.as_dict()), 4)

        # now re-mix the degenerate subspace of L_f
        for angle in (0.3, 1.1, -0.7):
            c, s = math.cos(angle), math.sin(angle)
            L_f2 = L_f.copy()
            L_f2[:, 2] = c * L_f[:, 2] + s * L_f[:, 3]
            L_f2[:, 3] = -s * L_f[:, 2] + c * L_f[:, 3]
            got = duschinsky.duschinsky_from_arrays(L_i, L_f2, mass, x_i, x_f,
                                                    freq_i, freq_f, verbose=0)
            # individual J elements in the block really do change
            self.assertGreater(abs(got.J[2:4] - ref.J[2:4]).max(), 1e-2)
            m_got = duschinsky.match_modes(got)
            self.assertEqual(m_got.blocks_f, m_ref.blocks_f)
            for a, b in zip(m_got, m_ref):
                self.assertEqual(a.degenerate, b.degenerate)
                self.assertAlmostEqual(a.overlap, b.overlap, 12)
                if a.degenerate:
                    self.assertAlmostEqual(
                        abs(a.block_singular_values - b.block_singular_values).max(),
                        0., 12)
            # the summed Huang-Rhys factor of the degenerate block is invariant
            self.assertAlmostEqual(got.huang_rhys[2:4].sum(),
                                   ref.huang_rhys[2:4].sum(), 12)

    def test_degenerate_block_from_real_pipeline(self):
        '''A genuinely degenerate pair produced by the HarmonicModel pipeline is
        detected as a block.'''
        coords = nonplanar_coords()
        charges = numpy.array([6, 1, 8, 7])
        mass_amu = numpy.array([12.011, 1.008, 15.9994, 14.007])
        freq_i = numpy.array([0.004, 0.007, 0.010, 0.013, 0.016, 0.019])
        freq_f = numpy.array([0.004, 0.007, 0.012, 0.012, 0.016, 0.019])
        st_i = make_state(coords, charges, mass_amu, freq_i, seed=91)
        st_f = make_state(coords, charges, mass_amu, freq_f, seed=92)
        self.assertLess(abs(st_f.model.freq[3] - st_f.model.freq[2]),
                        duschinsky.DEGENERACY_TOL)
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        m = d.match_modes()
        self.assertEqual([b for b in m.blocks_f if len(b) > 1], [[2, 3]])
        self.assertEqual(m.n_degenerate, 2)
        for rec in m:
            if rec.degenerate:
                self.assertIsNone(rec.mode_f)
                self.assertIn(rec.mode_f_raw, (2, 3))
                self.assertIn('degenerate', repr(rec))
                self.assertIn('meaningless', repr(rec))
            else:
                self.assertIn('ModeMatch', repr(rec))
        self.assertIn('ModeMatching', repr(m))
        self.assertEqual(len(m), d.nvib_i)
        self.assertEqual(m[0].mode_i, 0)

    # -- mode matching ----------------------------------------------------

    def test_match_modes_beats_greedy(self):
        '''The Hungarian assignment strictly beats greedy on a constructed J.

        With ``J**2 = [[0.81, 0.64], [0.64, 0.01]]`` greedy grabs the largest
        element ``(0,0)`` and is then forced onto ``(1,1)``, total ``0.82``.
        The optimum is the anti-diagonal, total ``1.28``.
        '''
        J = numpy.array([[0.9, 0.8], [0.8, 0.1]])
        freq = numpy.array([0.005, 0.010])
        m = duschinsky.match_modes(J, freq_i=freq, freq_f=freq)
        pairs = sorted((rec.mode_i, rec.mode_f) for rec in m)
        self.assertEqual(pairs, [(0, 1), (1, 0)])
        self.assertAlmostEqual(m.total_overlap, 1.28, 12)

        greedy_pairs, greedy_total = greedy_assignment(J**2)
        self.assertAlmostEqual(greedy_total, 0.82, 12)
        self.assertLess(greedy_total, m.total_overlap)
        self.assertNotEqual(sorted(greedy_pairs), sorted((rec.mode_f_raw, rec.mode_i)
                                                         for rec in m))

        # a larger case: the classic 3x3 where greedy is again suboptimal
        w = numpy.array([[0.90, 0.85, 0.10],
                         [0.85, 0.05, 0.05],
                         [0.10, 0.05, 0.80]])
        J3 = numpy.sqrt(w)
        f3 = numpy.array([0.004, 0.008, 0.012])
        m3 = duschinsky.match_modes(J3, freq_i=f3, freq_f=f3)
        _, greedy3 = greedy_assignment(w)
        self.assertAlmostEqual(m3.total_overlap, 0.85 + 0.85 + 0.80, 12)
        self.assertAlmostEqual(greedy3, 0.90 + 0.05 + 0.80, 12)
        self.assertGreater(m3.total_overlap, greedy3)

    def test_match_modes_identity(self):
        '''J = 1 -> the identity mapping, no degenerate flags.'''
        st = make_state(water_coords(), WATER_CHARGES, WATER_MASS, WATER_FREQ, seed=101)
        d = duschinsky.duschinsky_transform(st.model, st.model, verbose=0)
        m = d.match_modes()
        self.assertEqual(m.as_dict(), {0: 0, 1: 1, 2: 2})
        self.assertEqual(m.n_degenerate, 0)
        self.assertAlmostEqual(m.total_overlap, 3., 10)
        for rec in m:
            self.assertAlmostEqual(rec.overlap, 1., 10)
            self.assertAlmostEqual(rec.freq_shift, 0., 14)
        # dump() must not raise
        out = io.StringIO()
        m.stdout = out
        m.dump(verbose=lib.logger.NOTE)
        self.assertIn('mode_i', out.getvalue())

    def test_match_modes_by_overlap_not_frequency(self):
        '''When the frequency ordering and the overlap ordering disagree, the
        overlap wins.'''
        # J swaps modes 0 and 1 while the frequencies are ordered ascending in
        # both states: a frequency-based matcher would return the identity.
        J = numpy.array([[0.02, 0.9998, 0.], [0.9998, -0.02, 0.], [0., 0., 1.]])
        freq_i = numpy.array([0.004, 0.008, 0.012])
        freq_f = numpy.array([0.005, 0.009, 0.013])
        m = duschinsky.match_modes(J, freq_i=freq_i, freq_f=freq_f)
        self.assertEqual(m.as_dict(), {0: 1, 1: 0, 2: 2})

    def test_degeneracy_blocks_helper(self):
        blk = duschinsky.degeneracy_blocks(
            numpy.array([1., 1. + 1e-12, 2., 3., 3. + 1e-13, 3. + 2e-13]), 1e-9)
        self.assertEqual(blk, [[0, 1], [2], [3, 4, 5]])
        self.assertEqual(duschinsky.degeneracy_blocks(numpy.zeros(0), 1e-9), [])
        self.assertRaises(ValueError, duschinsky.degeneracy_blocks,
                          numpy.zeros(2), -1.)

    # -- diagnostics ------------------------------------------------------

    def test_excluded_mode_norm_small_when_aligned(self):
        '''A well-aligned, non-planar pair leaves almost nothing outside the
        final-state vibrational subspace.'''
        coords_i = nonplanar_coords()
        rs = numpy.random.RandomState(111)
        coords_f = coords_i + 0.02 * rs.randn(4, 3)
        charges = numpy.array([6, 1, 8, 7])
        mass_amu = numpy.array([12.011, 1.008, 15.9994, 14.007])
        freq = numpy.array([0.004, 0.007, 0.010, 0.013, 0.016, 0.019])
        st_i = make_state(coords_i, charges, mass_amu, freq, seed=112)
        st_f = make_state(coords_f, charges, mass_amu, freq * 0.95, seed=113)
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        diag = d.diagnostics
        self.assertGreater(diag['displacement_norm'], 1.)
        self.assertLess(diag['excluded_mode_norm'], duschinsky.EXCLUDED_NORM_TOL)
        self.assertLess(diag['displacement_reconstruction_error'], 1e-12)
        # the alignment really did enforce the Eckart rotational condition
        self.assertLess(diag['eckart_residual_after'],
                        1e-12 * diag['eckart_residual_scale'])
        self.assertGreater(diag['eckart_residual_before'],
                           1e-6 * diag['eckart_residual_scale'])
        self.assertGreater(diag['subspace_overlap_min'], 0.99)
        self.assertIsNotNone(diag['det_J'])
        for key in ('orthogonality_error', 'row_orthogonality_error', 'det_J',
                    'subspace_overlap', 'excluded_mode_norm',
                    'displacement_reconstruction_error', 'eckart_residual_before',
                    'eckart_residual_after', 'max_offdiag_J', 'mode_mixing',
                    'total_reorganization_energy', 'total_reorganization_energy_cm'):
            self.assertIn(key, diag)

    def test_excluded_mode_norm_detects_misalignment(self):
        '''With ``align=False`` and a rotated final geometry the diagnostic
        blows up and a warning is emitted.'''
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x_i = alignment.shift_to_center_of_mass(mass, water_coords(0.989, 0.989, 100.))
        x_f = alignment.shift_to_center_of_mass(mass, water_coords(1.04, 0.96, 108.))
        L_i = vib_basis(mass, x_i).dot(random_orthogonal(3, 121))
        L_f = vib_basis(mass, x_f).dot(random_orthogonal(3, 122))
        rot = random_rotation(9)

        ok = duschinsky.duschinsky_from_arrays(L_i, duschinsky.rotate_modes(L_f, rot),
                                               mass, x_i, x_f.dot(rot),
                                               WATER_FREQ, WATER_FREQ, verbose=0)
        self.assertLess(ok.diagnostics['excluded_mode_norm'],
                        duschinsky.EXCLUDED_NORM_TOL)

        out = io.StringIO()
        bad = duschinsky.duschinsky_from_arrays(
            L_i, duschinsky.rotate_modes(L_f, rot), mass, x_i, x_f.dot(rot),
            WATER_FREQ, WATER_FREQ, align=False, verbose=lib.logger.WARN, stdout=out)
        text = out.getvalue()
        self.assertGreater(bad.diagnostics['excluded_mode_norm'], 1e-2)
        self.assertIn('excluded_mode_norm', text)
        self.assertIn('WARN', text)
        self.assertGreater(bad.diagnostics['eckart_residual_after'],
                           1e-3 * bad.diagnostics['eckart_residual_scale'])
        self.assertFalse(bad.diagnostics['aligned'])
        # the spurious "displacement" is huge compared with the real one
        self.assertGreater(bad.diagnostics['displacement_norm'],
                           3 * ok.diagnostics['displacement_norm'])

    def test_unweighted_alignment_option(self):
        '''``mass_weighted_align=False`` is a *different*, non-Eckart frame; it
        must still run, be recorded, and leave the Eckart residual larger than
        the mass-weighted fit does.'''
        coords_i = nonplanar_coords()
        rs = numpy.random.RandomState(241)
        coords_f = coords_i + 0.05 * rs.randn(4, 3)
        charges = numpy.array([6, 1, 8, 7])
        mass_amu = numpy.array([12.011, 1.008, 15.9994, 14.007])
        freq = numpy.array([0.004, 0.007, 0.010, 0.013, 0.016, 0.019])
        st_i = make_state(coords_i, charges, mass_amu, freq, seed=242)
        st_f = make_state(coords_f, charges, mass_amu, freq * 0.95, seed=243)
        eck = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        geo = duschinsky.duschinsky_transform(st_i.model, st_f.model,
                                              mass_weighted_align=False, verbose=0)
        self.assertTrue(eck.diagnostics['mass_weighted_align'])
        self.assertFalse(geo.diagnostics['mass_weighted_align'])
        self.assertLess(eck.diagnostics['eckart_residual_after'],
                        geo.diagnostics['eckart_residual_after'])
        # the unweighted fit leaves more of the displacement outside the
        # vibrational subspace than the (correct) Eckart fit
        self.assertLessEqual(eck.diagnostics['excluded_mode_norm'],
                             geo.diagnostics['excluded_mode_norm'])
        assert_displacement_sign(self, geo, projected=True)

    def test_dump_and_repr(self):
        st_i = make_state(water_coords(0.989, 0.989, 100.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ, seed=131)
        st_f = make_state(water_coords(1.04, 0.96, 108.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ * 0.9, seed=132)
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        self.assertIn('Duschinsky', repr(d))
        out = io.StringIO()
        d.stdout = out
        d.dump_diagnostics(verbose=lib.logger.NOTE)
        text = out.getvalue()
        self.assertIn('total reorganization energy', text)
        self.assertIn('excluded_mode_norm', text)

    # -- inverse ----------------------------------------------------------

    def test_inverse_round_trip(self):
        st_i = make_state(water_coords(0.989, 0.989, 100.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ, seed=141)
        st_f = make_state(water_coords(1.04, 0.96, 108.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ * 0.9, seed=142)
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        inv = d.inverse()
        rs = numpy.random.RandomState(143)
        for _ in range(5):
            Q_i = rs.randn(d.nvib_i)
            self.assertAlmostEqual(abs(inv.apply(d.apply(Q_i)) - Q_i).max(), 0., 10)
            Q_f = rs.randn(d.nvib_f)
            self.assertAlmostEqual(abs(d.apply(inv.apply(Q_f)) - Q_f).max(), 0., 10)
        # roles swapped
        self.assertAlmostEqual(abs(inv.freq_i - d.freq_f).max(), 0., 14)
        self.assertAlmostEqual(abs(inv.freq_f - d.freq_i).max(), 0., 14)
        self.assertIs(inv.model_i, d.model_f)
        self.assertIs(inv.model_f, d.model_i)
        # J_rev = pinv(J) approaches J^T, and K_rev approaches the physical
        # -L_i^T M^{1/2}(x_i0 - x_f0) = -J^T K, to within the amount by which J
        # fails to be orthogonal (REFERENCES.md sec. 2.3).  This geometry change
        # is large enough that ||J^T J - 1|| ~ 4e-3, so the deviation is bounded
        # by a small multiple of that -- not by machine precision.
        orth = d.diagnostics['orthogonality_error']
        self.assertGreater(orth, 1e-4)               # genuinely non-orthogonal here
        self.assertLess(abs(inv.J - d.J.T).max(), 10 * orth)
        self.assertLess(abs(inv.K + d.J.T.dot(d.K)).max(), 10 * orth * abs(d.K).max())
        self.assertLess(abs(inv.K + d.modes_i.T.dot(d.displacement)).max(),
                        10 * orth * abs(d.K).max())
        self.assertLess(inv.diagnostics['inverse_physical_deviation_J'], 10 * orth)
        # inverse of the inverse recovers the original affine map exactly
        self.assertAlmostEqual(abs(inv.inverse().J - d.J).max(), 0., 10)
        self.assertAlmostEqual(abs(inv.inverse().K - d.K).max(), 0., 10)

    def test_inverse_exact_for_orthogonal_J(self):
        '''Two states at the same geometry -> J exactly orthogonal -> the
        pseudo-inverse *is* the transpose, to machine precision.'''
        coords = water_coords()
        st_i = make_state(coords, WATER_CHARGES, WATER_MASS, WATER_FREQ, seed=151)
        st_f = make_state(coords, WATER_CHARGES, WATER_MASS, WATER_FREQ * 0.8, seed=152)
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        self.assertAlmostEqual(d.diagnostics['orthogonality_error'], 0., 12)
        inv = d.inverse()
        self.assertAlmostEqual(abs(inv.J - d.J.T).max(), 0., 13)
        self.assertAlmostEqual(abs(inv.K + d.J.T.dot(d.K)).max(), 0., 12)
        self.assertAlmostEqual(abs(inv.K + d.modes_i.T.dot(d.displacement)).max(), 0., 12)
        self.assertLess(inv.diagnostics['inverse_physical_deviation_J'], 1e-13)
        self.assertLess(inv.diagnostics['inverse_physical_deviation_K'], 1e-11)

        # a displaced pair at the same geometry: J stays exactly orthogonal and
        # the physical reverse displacement is reproduced to machine precision
        st_g = make_state(alignment.shift_to_center_of_mass(
            numpy.asarray(units.amu2au(WATER_MASS)), coords), WATER_CHARGES,
            WATER_MASS, WATER_FREQ * 1.2, seed=153)
        d2 = duschinsky.duschinsky_transform(st_i.model, st_g.model, verbose=0)
        self.assertAlmostEqual(abs(d2.inverse().J - d2.J.T).max(), 0., 13)

    # -- error paths ------------------------------------------------------

    def test_error_different_natm(self):
        st = make_state(water_coords(), WATER_CHARGES, WATER_MASS, WATER_FREQ, seed=161)
        m_amu = numpy.array([1.008, 18.998403])
        hess = numpy.zeros((6, 6))
        hess[2, 2] = hess[5, 5] = 0.5
        hess[2, 5] = hess[5, 2] = -0.5
        di = normal_modes.HarmonicModel([1, 9], numpy.array([[0., 0., 0.], [0., 0., 1.8]]),
                                        m_amu, hess, verbose=0)
        with self.assertRaises(ValueError) as cm:
            duschinsky.duschinsky_transform(st.model, di, verbose=0)
        self.assertIn('different numbers of atoms', str(cm.exception))

    def test_error_different_charges(self):
        coords = water_coords()
        st_i = make_state(coords, [8, 1, 1], WATER_MASS, WATER_FREQ, seed=171)
        st_f = make_state(coords, [8, 1, 2], WATER_MASS, WATER_FREQ, seed=172)
        with self.assertRaises(ValueError) as cm:
            duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        self.assertIn('nuclear charges', str(cm.exception))

    def test_error_different_masses(self):
        '''An isotopic mismatch between the two states is a user error.'''
        coords = water_coords()
        st_i = make_state(coords, WATER_CHARGES, [15.9994, 1.008, 1.008],
                          WATER_FREQ, seed=181)
        st_f = make_state(coords, WATER_CHARGES, [15.9994, 2.0141, 1.008],
                          WATER_FREQ, seed=182)
        with self.assertRaises(ValueError) as cm:
            duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        self.assertIn('different atomic masses', str(cm.exception))
        # a tiny difference is tolerated, and the tolerance is adjustable
        st_f2 = make_state(coords, WATER_CHARGES, [15.9994, 1.008 * (1 + 1e-12), 1.008],
                           WATER_FREQ, seed=182)
        duschinsky.duschinsky_transform(st_i.model, st_f2.model, verbose=0)
        with self.assertRaises(ValueError):
            duschinsky.duschinsky_transform(st_i.model, st_f2.model, mass_rtol=1e-14,
                                            verbose=0)

    def test_error_different_rotor_type(self):
        '''REGULAR -> LINEAR changes nvib and is refused with an explanation.'''
        charges = [8, 1, 1]
        bent = make_state(water_coords(), charges, WATER_MASS, WATER_FREQ, seed=191)
        lin_coords = numpy.array([[0., 0., 0.], [0., 0., 1.8], [0., 0., -1.8]])
        lin = make_state(lin_coords, charges, WATER_MASS,
                         numpy.array([0.006, 0.006, 0.014, 0.017]), seed=192)
        self.assertEqual(bent.model.rotor_type, 'REGULAR')
        self.assertEqual(lin.model.rotor_type, 'LINEAR')
        self.assertEqual(bent.model.nvib, 3)
        self.assertEqual(lin.model.nvib, 4)
        with self.assertRaises(ValueError) as cm:
            duschinsky.duschinsky_transform(bent.model, lin.model, verbose=0)
        msg = str(cm.exception)
        self.assertIn('different rotor types', msg)
        self.assertIn('LINEAR', msg)
        # the message must explain *why*
        self.assertIn('nvib', msg)
        self.assertIn('curvilinear', msg)
        self.assertRaises(ValueError, duschinsky.duschinsky_transform,
                          lin.model, bent.model, verbose=0)

    def test_error_imaginary_frequency(self):
        coords_i = water_coords()
        st_i = make_state(coords_i, WATER_CHARGES, WATER_MASS, WATER_FREQ, seed=201)
        st_f = make_state(water_coords(1.04, 0.96, 108.), WATER_CHARGES, WATER_MASS,
                          WATER_FREQ, seed=202, imaginary=(0,),
                          imaginary_policy='warn')
        self.assertTrue(st_f.model.imaginary.any())
        self.assertLess(st_f.model.freq[0], 0)
        with self.assertRaises(ValueError) as cm:
            duschinsky.duschinsky_transform(st_i.model, st_f.model, verbose=0)
        self.assertIn('imaginary', str(cm.exception))
        # ... and in the initial state too
        self.assertRaises(ValueError, duschinsky.duschinsky_transform,
                          st_f.model, st_i.model, verbose=0)
        # opt-in: proceeds, warns, and marks the affected mode nan
        out = io.StringIO()
        st_i.model.stdout = out
        d = duschinsky.duschinsky_transform(st_i.model, st_f.model,
                                            allow_imaginary=True,
                                            verbose=lib.logger.WARN)
        self.assertIn('imaginary frequencies found', out.getvalue())
        self.assertTrue(numpy.isnan(d.huang_rhys[0]))
        self.assertFalse(numpy.isnan(d.huang_rhys[1:]).any())
        self.assertTrue(numpy.isfinite(d.total_reorganization_energy))

    def test_error_bad_input(self):
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x = alignment.shift_to_center_of_mass(mass, water_coords())
        L = vib_basis(mass, x).dot(random_orthogonal(3, 211))
        kw = dict(verbose=0)
        # non-orthonormal L
        self.assertRaises(ValueError, duschinsky.duschinsky_from_arrays,
                          L * 2, L, mass, x, x, WATER_FREQ, WATER_FREQ, **kw)
        self.assertRaises(ValueError, duschinsky.duschinsky_from_arrays,
                          L, L * 2, mass, x, x, WATER_FREQ, WATER_FREQ, **kw)
        # wrong number of frequencies
        self.assertRaises(ValueError, duschinsky.duschinsky_from_arrays,
                          L, L, mass, x, x, WATER_FREQ[:2], WATER_FREQ, **kw)
        # wrong row count
        self.assertRaises(ValueError, duschinsky.duschinsky_from_arrays,
                          L[:6], L, mass, x, x, WATER_FREQ, WATER_FREQ, **kw)
        # bad mass unit
        self.assertRaises(ValueError, duschinsky.duschinsky_from_arrays,
                          L, L, mass, x, x, WATER_FREQ, WATER_FREQ,
                          mass_unit='kg', **kw)
        # rotate_modes input validation
        self.assertRaises(ValueError, duschinsky.rotate_modes, L, numpy.eye(2))
        self.assertRaises(ValueError, duschinsky.rotate_modes, L[:8], numpy.eye(3))
        # not a HarmonicModel
        self.assertRaises(ValueError, duschinsky.duschinsky_transform,
                          object(), object())
        # match_modes shape validation
        self.assertRaises(ValueError, duschinsky.match_modes, numpy.zeros(3))
        self.assertRaises(ValueError, duschinsky.match_modes, numpy.eye(3),
                          WATER_FREQ[:2], WATER_FREQ)

    def test_atom_no_vibrations(self):
        '''An atom has no vibrational degrees of freedom; nothing must crash and
        every diagnostic must be well defined.'''
        m = normal_modes.HarmonicModel([8], [[0., 0., 0.]], [15.9994],
                                       numpy.zeros((3, 3)), verbose=0)
        self.assertEqual(m.rotor_type, 'ATOM')
        self.assertEqual(m.nvib, 0)
        d = duschinsky.duschinsky_transform(m, m, verbose=0)
        self.assertEqual(d.J.shape, (0, 0))
        self.assertEqual(d.K.shape, (0,))
        self.assertIsNone(d.diagnostics['det_J'])
        self.assertEqual(d.total_reorganization_energy, 0.)
        self.assertEqual(len(d.match_modes()), 0)
        self.assertEqual(d.inverse().J.shape, (0, 0))
        out = io.StringIO()
        d.stdout = out
        d.dump_diagnostics(verbose=lib.logger.NOTE)
        self.assertIn('nvib_i = 0', out.getvalue())

    def test_rotate_modes_preserves_orthonormality(self):
        mass = numpy.asarray(units.amu2au(WATER_MASS))
        x = alignment.shift_to_center_of_mass(mass, water_coords())
        L = vib_basis(mass, x).dot(random_orthogonal(3, 221))
        for seed in range(4):
            Lr = duschinsky.rotate_modes(L, random_rotation(seed))
            self.assertAlmostEqual(abs(Lr.T.dot(Lr) - numpy.eye(3)).max(), 0., 14)
        # rotating by R then R^T is the identity
        rot = random_rotation(5)
        self.assertAlmostEqual(
            abs(duschinsky.rotate_modes(duschinsky.rotate_modes(L, rot), rot.T)
                - L).max(), 0., 14)

    # -- ab initio ---------------------------------------------------------

    def test_real_water(self):
        '''RHF/STO-3G water at two displaced geometries.

        Initial: r(OH) = 0.989/0.989 A, HOH = 100 deg.
        Final:   r(OH) = 1.060/0.950 A, HOH = 112 deg.

        Making the two OH bonds inequivalent switches the normal modes from
        symmetric/antisymmetric stretches to nearly localised OH stretches, so
        there is large, genuine mode mixing: ``max|J_offdiag| ~ 0.61``.

        Tolerance on ``||J^T J - 1||``.  ``J`` is exactly orthogonal only when
        the two states span the same vibrational subspace, i.e. when the six
        translation/rotation vectors coincide.  Those vectors are built at each
        state's *own* geometry, so they genuinely differ once the geometries do,
        and the deviation grows linearly with the geometry change.  Here it is
        ~1.0e-2 for a ~7% bond-length change and a 12 deg angle change; the
        module warns above ``ORTHOGONALITY_TOL`` = 1e-3 and that warning is
        expected and correct for a change this large.  The bound asserted below
        (2e-2) is set from that estimate, not tuned to pass.
        '''
        mi, mf = _REAL['i'], _REAL['f']
        self.assertEqual(mi.rotor_type, 'REGULAR')
        self.assertEqual(mi.nvib, 3)
        self.assertFalse(mi.imaginary.any())
        self.assertFalse(mf.imaginary.any())
        d = duschinsky.duschinsky_transform(mi, mf, verbose=0)
        diag = d.diagnostics

        # real mode mixing, but still recognisably a rotation of the identity
        self.assertGreater(diag['max_offdiag_J'], 0.5)
        self.assertGreater(abs(d.J).max(), 0.9)
        self.assertGreater(abs(d.J - numpy.eye(3)).max(), 0.1)

        self.assertLess(diag['orthogonality_error'], 2e-2)
        self.assertLess(diag['row_orthogonality_error'], 2e-2)
        self.assertLess(abs(abs(diag['det_J']) - 1.), 2e-2)
        self.assertGreater(diag['subspace_overlap_min'], 0.98)
        self.assertLess(diag['subspace_overlap_max'], 1. + 1e-10)

        # both geometries are planar and share the plane, so M^{1/2}(x_i0-x_f0)
        # lies entirely in the in-plane vibrational subspace after alignment
        self.assertLess(diag['excluded_mode_norm'], 1e-12)
        self.assertLess(diag['displacement_reconstruction_error'], 1e-12)
        self.assertLess(diag['eckart_residual_after'],
                        1e-12 * diag['eckart_residual_scale'])
        self.assertLess(diag['modes_orthonormality_error_i'], 1e-12)
        self.assertLess(diag['modes_orthonormality_error_f'], 1e-12)

        # Huang-Rhys factors and the reorganisation energy are O(0.1-1) and
        # consistent with each other
        self.assertTrue(numpy.all(d.huang_rhys >= 0))
        self.assertGreater(d.huang_rhys.max(), 0.1)
        self.assertAlmostEqual(
            abs(d.reorganization_energy - 0.5 * d.freq_f**2 * d.K**2).max(), 0., 14)
        self.assertAlmostEqual(d.total_reorganization_energy,
                               d.reorganization_energy.sum(), 14)

        # the executed relation Q_f = J Q_i + K
        rs = numpy.random.RandomState(231)
        Q_i = rs.randn(3)
        ref = d.modes_f.T.dot(d.modes_i.dot(Q_i) + d.displacement)
        self.assertAlmostEqual(abs(d.apply(Q_i) - ref).max(), 0., 12)
        self.assertAlmostEqual(abs(d.inverse().apply(d.apply(Q_i)) - Q_i).max(), 0., 9)

        # mode correlation: no degeneracies here, so a clean 1-1 mapping
        m = d.match_modes()
        self.assertEqual(m.n_degenerate, 0)
        self.assertEqual(sorted(m.as_dict().values()), [0, 1, 2])


if __name__ == '__main__':
    print('Tests for pyscf.vibronic.duschinsky')
    unittest.main()
