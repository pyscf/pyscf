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

from pyscf.vibronic import alignment


def random_rotation(rng):
    '''A Haar-ish random proper rotation via QR of a Gaussian matrix.'''
    a = rng.standard_normal((3, 3))
    q, r = numpy.linalg.qr(a)
    q = q * numpy.sign(numpy.diag(r))       # make the decomposition unique
    if numpy.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# Geometries in bohr.  Masses in amu (the alignment module is unit-agnostic).
GEOM = {
    # H2O, near the experimental structure
    'H2O': (numpy.array([15.99491, 1.00783, 1.00783]),
            numpy.array([[0.000000,  0.000000,  0.221665],
                         [0.000000,  1.430919, -0.886661],
                         [0.000000, -1.430919, -0.886661]])),
    'H2': (numpy.array([1.00783, 1.00783]),
           numpy.array([[0., 0., 0.], [0., 0., 1.4]])),
    'CO2': (numpy.array([12.0, 15.99491, 15.99491]),
            numpy.array([[0., 0., 0.], [0., 0., 2.2], [0., 0., -2.2]])),
    'NH3': (numpy.array([14.00307, 1.00783, 1.00783, 1.00783]),
            numpy.array([[0.000000,  0.000000,  0.128000],
                         [0.000000,  1.771000, -0.298000],
                         [1.533700, -0.885500, -0.298000],
                         [-1.533700, -0.885500, -0.298000]])),
    'CH4': (numpy.array([12.0, 1.00783, 1.00783, 1.00783, 1.00783]),
            numpy.array([[0.000000,  0.000000,  0.000000],
                         [1.185000,  1.185000,  1.185000],
                         [-1.185000, -1.185000,  1.185000],
                         [-1.185000,  1.185000, -1.185000],
                         [1.185000, -1.185000, -1.185000]])),
    'ATOM': (numpy.array([4.0026]), numpy.zeros((1, 3))),
}

# A genuinely chiral 4-atom skeleton (a CHFClBr-like tetrahedron with four
# different substituent distances), so that its mirror image cannot be
# superimposed on it by any proper rotation.
CHIRAL_MASS = numpy.array([12.0, 1.00783, 18.99840, 34.96885])
CHIRAL_COORDS = numpy.array([[0.00, 0.00, 0.00],
                             [1.90, 1.90, 1.90],
                             [-2.40, -2.40, 1.70],
                             [-3.10, 3.30, -2.90]])


def bent_co2(eps, d=2.2):
    '''CO2 with each O displaced off-axis by an angle ``eps`` (radians).'''
    return numpy.array([[0., 0., 0.],
                        [d * numpy.cos(eps), d * numpy.sin(eps), 0.],
                        [-d * numpy.cos(eps), d * numpy.sin(eps), 0.]])


class KnownValues(unittest.TestCase):

    # -- centring and inertia ----------------------------------------------

    def test_center_of_mass(self):
        mass, coords = GEOM['H2O']
        com = alignment.center_of_mass(mass, coords)
        ref = (mass[:, None] * coords).sum(axis=0) / mass.sum()
        self.assertAlmostEqual(abs(com - ref).max(), 0, 14)
        centred = alignment.shift_to_center_of_mass(mass, coords)
        self.assertAlmostEqual(abs(alignment.center_of_mass(mass, centred)).max(), 0, 14)
        # translating the input must not change the centred structure
        shifted = alignment.shift_to_center_of_mass(mass, coords + numpy.array([3., -1., 7.]))
        self.assertAlmostEqual(abs(shifted - centred).max(), 0, 13)

    def test_inertia_tensor_translation_invariant(self):
        mass, coords = GEOM['NH3']
        i0 = alignment.inertia_tensor(mass, coords)
        i1 = alignment.inertia_tensor(mass, coords + numpy.array([10., -5., 2.]))
        self.assertAlmostEqual(abs(i0 - i1).max(), 0, 11)
        self.assertAlmostEqual(abs(i0 - i0.T).max(), 0, 14)

    def test_inertia_tensor_rotation_covariant(self):
        rng = numpy.random.default_rng(7)
        mass, coords = GEOM['CH4']
        rot = random_rotation(rng)
        i0 = alignment.inertia_tensor(mass, coords)
        i1 = alignment.inertia_tensor(mass, coords.dot(rot))
        # I -> R^T I R
        self.assertAlmostEqual(abs(i1 - rot.T.dot(i0).dot(rot)).max(), 0, 10)
        # principal moments are invariant
        self.assertAlmostEqual(abs(alignment.principal_moments(mass, coords)
                                   - alignment.principal_moments(mass, coords.dot(rot))).max(),
                               0, 10)

    def test_rotation_constants(self):
        mass, coords = GEOM['H2O']
        b = alignment.rotation_constants(mass, coords)
        moments = alignment.principal_moments(mass, coords)
        self.assertTrue(numpy.all(numpy.diff(b) <= 0))   # descending A >= B >= C
        self.assertAlmostEqual(abs(numpy.sort(b) - numpy.sort(1. / (2 * moments))).max(), 0, 12)
        # linear molecule -> one infinite constant
        mass, coords = GEOM['CO2']
        b = alignment.rotation_constants(mass, coords)
        self.assertTrue(numpy.isinf(b[0]))
        self.assertAlmostEqual(b[1], b[2], 12)

    # -- rotor classification ----------------------------------------------

    def test_classify_rotor(self):
        self.assertEqual(alignment.classify_rotor(*GEOM['ATOM']), 'ATOM')
        self.assertEqual(alignment.classify_rotor(*GEOM['H2']), 'LINEAR')
        self.assertEqual(alignment.classify_rotor(*GEOM['CO2']), 'LINEAR')
        self.assertEqual(alignment.classify_rotor(*GEOM['H2O']), 'REGULAR')
        self.assertEqual(alignment.classify_rotor(*GEOM['NH3']), 'REGULAR')
        self.assertEqual(alignment.classify_rotor(*GEOM['CH4']), 'REGULAR')
        # HCN, a linear molecule with no centre of symmetry
        hcn_m = numpy.array([1.00783, 12.0, 14.00307])
        hcn_c = numpy.array([[0., 0., 0.], [0., 0., 2.0], [0., 0., 4.2]])
        self.assertEqual(alignment.classify_rotor(hcn_m, hcn_c), 'LINEAR')

    def test_classify_rotor_scale_invariance(self):
        '''The criterion is relative, so it must survive a uniform rescaling of
        the coordinates or the masses (an absolute threshold would not).'''
        mass, coords = GEOM['CO2']
        for scale in (1e-3, 1.0, 1e3):
            self.assertEqual(alignment.classify_rotor(mass, coords * scale), 'LINEAR')
            self.assertEqual(alignment.classify_rotor(mass * scale, coords), 'LINEAR')
        mass, coords = GEOM['H2O']
        for scale in (1e-3, 1.0, 1e3):
            self.assertEqual(alignment.classify_rotor(mass, coords * scale), 'REGULAR')
            self.assertEqual(alignment.classify_rotor(mass * scale, coords), 'REGULAR')

    def test_classify_rotor_orientation_invariance(self):
        rng = numpy.random.default_rng(11)
        mass, coords = GEOM['CO2']
        for _ in range(5):
            rot = random_rotation(rng)
            shift = rng.standard_normal(3) * 5
            self.assertEqual(alignment.classify_rotor(mass, coords.dot(rot) + shift), 'LINEAR')

    def test_bent_co2_is_regular(self):
        mass = GEOM['CO2'][0]
        # 0.5 degrees is well beyond the resolution boundary (see below)
        self.assertEqual(alignment.classify_rotor(mass, bent_co2(numpy.radians(0.5))), 'REGULAR')
        self.assertEqual(alignment.classify_rotor(mass, bent_co2(numpy.radians(30.))), 'REGULAR')

    def test_near_linear_boundary(self):
        '''Pin down where the LINEAR/REGULAR boundary actually lies.

        For a bent A-B-A triatomic, I0/I2 = c*eps**2 with c a geometry-dependent
        constant (0.2728 for this CO2), so the default lin_tol=1e-6 puts the
        boundary at eps = sqrt(1e-6/0.2728) = 1.914e-3 rad = 0.1097 degrees.
        Anything smaller is (deliberately) treated as exactly linear.
        '''
        mass = GEOM['CO2'][0]
        eps_small = 1e-4
        ratio = (alignment.principal_moments(mass, bent_co2(eps_small))[0]
                 / alignment.principal_moments(mass, bent_co2(eps_small))[2])
        c = ratio / eps_small**2
        self.assertAlmostEqual(c, 0.2728, 4)

        eps_boundary = numpy.sqrt(alignment.LINEAR_TOL / c)
        self.assertAlmostEqual(eps_boundary, 1.9145e-3, 6)
        self.assertAlmostEqual(numpy.degrees(eps_boundary), 0.10969, 4)

        # bracket the boundary tightly: 1% below -> LINEAR, 1% above -> REGULAR
        self.assertEqual(alignment.classify_rotor(mass, bent_co2(eps_boundary * 0.99)), 'LINEAR')
        self.assertEqual(alignment.classify_rotor(mass, bent_co2(eps_boundary * 1.01)), 'REGULAR')

        # and the tolerance really is a knob
        self.assertEqual(alignment.classify_rotor(mass, bent_co2(1e-2), lin_tol=1e-3), 'LINEAR')
        self.assertEqual(alignment.classify_rotor(mass, bent_co2(1e-4), lin_tol=1e-12), 'REGULAR')

    def test_classify_rotor_bad_input(self):
        mass, coords = GEOM['H2O']
        self.assertRaises(ValueError, alignment.classify_rotor, mass, coords, -1.)
        self.assertRaises(ValueError, alignment.classify_rotor, mass[:2], coords)
        self.assertRaises(ValueError, alignment.classify_rotor, [-1., 1., 1.], coords)
        bad = coords.copy()
        bad[0, 0] = numpy.nan
        self.assertRaises(ValueError, alignment.classify_rotor, mass, bad)

    # -- Kabsch ------------------------------------------------------------

    def test_kabsch_recovers_random_rotation(self):
        rng = numpy.random.default_rng(2024)
        mass, coords = GEOM['CH4']
        ref = alignment.shift_to_center_of_mass(mass, coords)
        for _ in range(20):
            rot = random_rotation(rng)
            rotated = ref.dot(rot)
            # recover: we want R_fit with rotated @ R_fit == ref, i.e. R_fit = rot.T
            rfit = alignment.kabsch_rotation(ref, rotated, weights=mass)
            self.assertAlmostEqual(abs(rfit - rot.T).max(), 0, 10)
            self.assertAlmostEqual(numpy.linalg.det(rfit), 1.0, 12)
            self.assertAlmostEqual(alignment.rmsd(ref, rotated.dot(rfit), mass), 0, 10)

    def test_kabsch_is_a_rotation(self):
        rng = numpy.random.default_rng(5)
        for natm in (1, 2, 3, 4, 8):
            mass = rng.uniform(1., 40., natm)
            a = rng.standard_normal((natm, 3)) * 2
            b = rng.standard_normal((natm, 3)) * 2
            a = alignment.shift_to_center_of_mass(mass, a)
            b = alignment.shift_to_center_of_mass(mass, b)
            rot = alignment.kabsch_rotation(a, b, weights=mass)
            self.assertAlmostEqual(abs(rot.dot(rot.T) - numpy.eye(3)).max(), 0, 12)
            self.assertAlmostEqual(numpy.linalg.det(rot), 1.0, 12)

    def test_kabsch_requires_centred_input(self):
        mass, coords = GEOM['H2O']
        ref = alignment.shift_to_center_of_mass(mass, coords)
        self.assertRaises(ValueError, alignment.kabsch_rotation, ref + 1e-3, ref, mass)
        self.assertRaises(ValueError, alignment.kabsch_rotation, ref, ref + 1e-3, mass)
        # within tolerance is fine
        alignment.kabsch_rotation(ref, ref + 1e-12, weights=mass)

    def test_kabsch_never_reflects(self):
        '''A chiral structure and its mirror image cannot be superimposed by a
        proper rotation.  Kabsch must return det(R) = +1 and a genuinely
        nonzero residual RMSD, not "cheat" by returning the reflection.'''
        mass = CHIRAL_MASS
        ref = alignment.shift_to_center_of_mass(mass, CHIRAL_COORDS)
        mirror = ref * numpy.array([1., 1., -1.])
        mirror = alignment.shift_to_center_of_mass(mass, mirror)

        rot = alignment.kabsch_rotation(ref, mirror, weights=mass)
        self.assertAlmostEqual(numpy.linalg.det(rot), 1.0, 12)
        self.assertAlmostEqual(abs(rot.dot(rot.T) - numpy.eye(3)).max(), 0, 12)

        # The residual is a substantial fraction of the molecular size: this is a
        # real geometric mismatch, not numerical noise.  (Measured: 0.5441 bohr,
        # 15.7% of the mass-weighted radius of gyration 3.4692 bohr.)
        r_gyr = numpy.sqrt(numpy.einsum('a,ax,ax->', mass, ref, ref) / mass.sum())
        resid = alignment.rmsd(ref, mirror.dot(rot), mass)
        self.assertAlmostEqual(r_gyr, 3.4692, 4)
        self.assertAlmostEqual(resid, 0.54406, 5)
        self.assertGreater(resid / r_gyr, 0.1)

        # ... and it really is the best a *proper* rotation can do
        rng = numpy.random.default_rng(1234)
        for _ in range(2000):
            trial = random_rotation(rng)
            self.assertGreaterEqual(alignment.rmsd(ref, mirror.dot(trial), mass), resid - 1e-12)

        # the improper "rotation" would fit perfectly -- confirm the test has teeth
        improper = numpy.diag([1., 1., -1.])
        self.assertAlmostEqual(numpy.linalg.det(improper), -1.0, 12)
        self.assertAlmostEqual(alignment.rmsd(ref, mirror.dot(improper), mass), 0, 12)

    def test_kabsch_minimises_rmsd(self):
        '''The returned rotation must beat random rotations, and be stationary.'''
        rng = numpy.random.default_rng(31)
        mass = CHIRAL_MASS
        ref = alignment.shift_to_center_of_mass(mass, CHIRAL_COORDS)
        perturbed = alignment.shift_to_center_of_mass(
            mass, CHIRAL_COORDS.dot(random_rotation(rng)) + rng.standard_normal((4, 3)) * 0.1)
        rot = alignment.kabsch_rotation(ref, perturbed, weights=mass)
        best = alignment.rmsd(ref, perturbed.dot(rot), mass)
        for _ in range(200):
            trial = rot.dot(random_rotation(rng))
            self.assertGreaterEqual(alignment.rmsd(ref, perturbed.dot(trial), mass), best - 1e-12)

    def test_mass_weighting_changes_the_answer(self):
        '''For an isotopically asymmetric system the mass-weighted and the
        equal-weight fits are genuinely different rotations.'''
        rng = numpy.random.default_rng(99)
        # HDO-like: same skeleton, very different masses
        mass = numpy.array([15.99491, 1.00783, 2.01410])
        coords = GEOM['H2O'][1]
        target = coords.dot(random_rotation(rng)) + rng.standard_normal((3, 3)) * 0.25

        _, rot_mw = alignment.align_geometries(mass, coords, target, mass_weighted=True)
        _, rot_eq = alignment.align_geometries(mass, coords, target, mass_weighted=False)
        self.assertGreater(abs(rot_mw - rot_eq).max(), 1e-3)

        # each is optimal in its own metric
        ref_mw = alignment.shift_to_center_of_mass(mass, coords)
        tgt = alignment.shift_to_center_of_mass(mass, target)
        self.assertLess(alignment.rmsd(ref_mw, tgt.dot(rot_mw), mass),
                        alignment.rmsd(ref_mw, tgt.dot(rot_eq), mass))
        self.assertLess(alignment.rmsd(ref_mw, tgt.dot(rot_eq)),
                        alignment.rmsd(ref_mw, tgt.dot(rot_mw)))

        # with equal masses the two coincide
        eq_mass = numpy.array([12., 12., 12.])
        _, r1 = alignment.align_geometries(eq_mass, coords, target, mass_weighted=True)
        _, r2 = alignment.align_geometries(eq_mass, coords, target, mass_weighted=False)
        self.assertAlmostEqual(abs(r1 - r2).max(), 0, 12)

    # -- Eckart ------------------------------------------------------------

    def test_eckart_residual(self):
        rng = numpy.random.default_rng(17)
        mass, coords = GEOM['NH3']
        # a displaced and rotated copy
        displaced = coords + rng.standard_normal(coords.shape) * 0.05
        misoriented = displaced.dot(random_rotation(rng)) + numpy.array([2., -1., 0.5])

        scale = numpy.einsum('a,ax,ax->', mass,
                             alignment.shift_to_center_of_mass(mass, coords),
                             alignment.shift_to_center_of_mass(mass, coords))
        before = alignment.eckart_residual(mass, coords, misoriented)
        self.assertGreater(before / scale, 0.1)

        aligned, _ = alignment.align_geometries(mass, coords, misoriented)
        after = alignment.eckart_residual(mass, coords, aligned)
        self.assertLess(after / scale, 1e-13)

    def test_eckart_residual_zero_for_identical(self):
        mass, coords = GEOM['CH4']
        self.assertAlmostEqual(alignment.eckart_residual(mass, coords, coords), 0, 11)
        # translation-invariant
        self.assertAlmostEqual(
            alignment.eckart_residual(mass, coords, coords + numpy.array([1., 2., 3.])), 0, 10)

    def test_align_geometries_roundtrip(self):
        rng = numpy.random.default_rng(3)
        mass, coords = GEOM['H2O']
        for _ in range(10):
            rot = random_rotation(rng)
            moved = coords.dot(rot) + rng.standard_normal(3) * 4
            aligned, rfit = alignment.align_geometries(mass, coords, moved)
            ref = alignment.shift_to_center_of_mass(mass, coords)
            self.assertAlmostEqual(abs(aligned - ref).max(), 0, 10)
            self.assertAlmostEqual(numpy.linalg.det(rfit), 1.0, 12)
            self.assertAlmostEqual(abs(alignment.center_of_mass(mass, aligned)).max(), 0, 12)

    def test_eckart_frame_alias(self):
        rng = numpy.random.default_rng(77)
        mass, coords = GEOM['NH3']
        moved = coords.dot(random_rotation(rng)) + rng.standard_normal(3)
        a1, r1 = alignment.align_geometries(mass, coords, moved)
        a2, r2 = alignment.eckart_frame(mass, coords, moved)
        self.assertAlmostEqual(abs(a1 - a2).max(), 0, 15)
        self.assertAlmostEqual(abs(r1 - r2).max(), 0, 15)

    def test_align_geometries_is_idempotent(self):
        rng = numpy.random.default_rng(23)
        mass, coords = GEOM['NH3']
        moved = coords.dot(random_rotation(rng)) + rng.standard_normal(coords.shape) * 0.1
        a1, _ = alignment.align_geometries(mass, coords, moved)
        a2, r2 = alignment.align_geometries(mass, coords, a1)
        self.assertAlmostEqual(abs(a2 - a1).max(), 0, 11)
        self.assertAlmostEqual(abs(r2 - numpy.eye(3)).max(), 0, 10)

    # -- degenerate cases ---------------------------------------------------

    def test_single_atom(self):
        mass, coords = GEOM['ATOM']
        centred = alignment.shift_to_center_of_mass(mass, coords + 5.)
        self.assertAlmostEqual(abs(centred).max(), 0, 14)
        rot = alignment.kabsch_rotation(centred, centred, weights=mass)
        self.assertAlmostEqual(numpy.linalg.det(rot), 1.0, 12)
        self.assertAlmostEqual(abs(rot.dot(rot.T) - numpy.eye(3)).max(), 0, 12)
        aligned, rot = alignment.align_geometries(mass, coords, coords + numpy.array([3., 4., 5.]))
        self.assertAlmostEqual(abs(aligned).max(), 0, 12)
        self.assertAlmostEqual(numpy.linalg.det(rot), 1.0, 12)
        self.assertAlmostEqual(alignment.eckart_residual(mass, coords, coords + 1.), 0, 14)

    def test_diatomic_rotation_underdetermined(self):
        '''For two atoms the rotation about the bond axis is undetermined; the
        SVD still returns a proper rotation and the aligned RMSD is exact.'''
        rng = numpy.random.default_rng(41)
        mass = numpy.array([1.00783, 34.96885])          # HCl, strongly asymmetric
        coords = numpy.array([[0., 0., 0.], [0., 0., 2.41]])
        for _ in range(10):
            moved = coords.dot(random_rotation(rng)) + rng.standard_normal(3)
            aligned, rot = alignment.align_geometries(mass, coords, moved)
            self.assertAlmostEqual(numpy.linalg.det(rot), 1.0, 12)
            self.assertAlmostEqual(abs(rot.dot(rot.T) - numpy.eye(3)).max(), 0, 12)
            ref = alignment.shift_to_center_of_mass(mass, coords)
            self.assertAlmostEqual(alignment.rmsd(ref, aligned, mass), 0, 10)
            self.assertLess(alignment.eckart_residual(mass, coords, aligned)
                            / numpy.einsum('a,ax,ax->', mass, ref, ref), 1e-13)

    def test_diatomic_stretched(self):
        '''Different bond lengths: the RMSD after alignment must equal the
        analytic mass-weighted value, and the atoms must not swap.'''
        mass = numpy.array([1.00783, 34.96885])
        a = numpy.array([[0., 0., 0.], [0., 0., 2.41]])
        b = numpy.array([[0., 0., 0.], [0., 0., 2.61]])
        aligned, rot = alignment.align_geometries(mass, a, b)
        ref = alignment.shift_to_center_of_mass(mass, a)
        # both are along +z after centring, so the residual is a pure stretch
        mu = mass[0] * mass[1] / mass.sum()
        expect = numpy.sqrt(mu * 0.20**2 / mass.sum())
        self.assertAlmostEqual(alignment.rmsd(ref, aligned, mass), expect, 12)
        self.assertAlmostEqual(numpy.linalg.det(rot), 1.0, 12)

    def test_rmsd_and_weights_validation(self):
        mass, coords = GEOM['H2O']
        self.assertAlmostEqual(alignment.rmsd(coords, coords), 0, 14)
        d = coords + 0.1
        self.assertAlmostEqual(alignment.rmsd(coords, d), numpy.sqrt(3 * 0.01), 12)
        ref = alignment.shift_to_center_of_mass(mass, coords)
        self.assertRaises(ValueError, alignment.kabsch_rotation, ref, ref, [1., 1.])
        self.assertRaises(ValueError, alignment.kabsch_rotation, ref, ref, [-1., 1., 1.])
        self.assertRaises(ValueError, alignment.kabsch_rotation, ref, ref, [0., 0., 0.])
        self.assertRaises(ValueError, alignment.kabsch_rotation, ref, ref[:2], mass)


if __name__ == '__main__':
    print('Full Tests for pyscf.vibronic.alignment')
    unittest.main()
