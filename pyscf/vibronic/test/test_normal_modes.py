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
from io import StringIO

from pyscf import gto, scf, lib
from pyscf.hessian import thermo
from pyscf.vibronic import normal_modes, alignment, units
from pyscf.vibronic.normal_modes import HarmonicModel

mol_h2o = None
hess_h2o = None
e_h2o = None


def setUpModule():
    global mol_h2o, hess_h2o, e_h2o
    mol_h2o = gto.M(atom='''O   0.0000000   0.0000000   0.1173000
                            H   0.0000000   0.7572000  -0.4692000
                            H   0.0000000  -0.7572000  -0.4692000''',
                    basis='sto-3g', verbose=0)
    mf = scf.RHF(mol_h2o).run()
    e_h2o = mf.e_tot
    hess_h2o = mf.Hessian().kernel()


def tearDownModule():
    global mol_h2o, hess_h2o, e_h2o
    del mol_h2o, hess_h2o, e_h2o


def random_rotation(rng):
    a = rng.standard_normal((3, 3))
    q, r = numpy.linalg.qr(a)
    q = q * numpy.sign(numpy.diag(r))
    if numpy.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def rotate_hessian(hess_3n3n, rot):
    '''Cartesian Hessian in the frame where ``coords -> coords @ R``.

    With ``x'_{a,c} = sum_d x_{a,d} R_{d,c}`` the second derivatives transform
    as ``H'_{ac,be} = sum_{d,f} R_{d,c} H_{ad,bf} R_{f,e}``.
    '''
    n3 = hess_3n3n.shape[0]
    natm = n3 // 3
    h4 = hess_3n3n.reshape(natm, 3, natm, 3)
    h4 = numpy.einsum('adbf,dc,fe->acbe', h4, rot, rot)
    return h4.reshape(n3, n3)


def psd_hessian(natm, seed, scale=0.05):
    '''A generic positive-semidefinite symmetric (3N,3N) matrix.

    Not a physical Hessian, but sufficient for counting vibrational degrees of
    freedom: the count depends only on the geometry and the rotor type.
    '''
    rng = numpy.random.default_rng(seed)
    a = rng.standard_normal((3 * natm, 3 * natm))
    return scale * a.dot(a.T)


def diatomic_stretch_hessian(k, direction=(0., 0., 1.)):
    '''Exact Cartesian Hessian of ``V = k/2 (r - r0)^2`` for a diatomic.

    ``H = k v v^T`` with ``v = [n, -n]`` and ``n`` the bond unit vector.  The
    single nonzero mass-weighted eigenvalue is ``k*(1/m1 + 1/m2) = k/mu``.
    '''
    n = numpy.asarray(direction, dtype=float)
    n = n / numpy.linalg.norm(n)
    v = numpy.concatenate([n, -n])
    return k * numpy.outer(v, v)


# geometries in bohr
CO2_COORDS = numpy.array([[0., 0., 0.], [0., 0., 2.2], [0., 0., -2.2]])
CO2_MASS = numpy.array([12.0, 15.99491, 15.99491])
CO2_Z = numpy.array([6, 8, 8])

HCN_COORDS = numpy.array([[0., 0., 0.], [0., 0., 2.0], [0., 0., 4.19]])
HCN_MASS = numpy.array([1.00783, 12.0, 14.00307])
HCN_Z = numpy.array([1, 6, 7])

NH3_COORDS = numpy.array([[0.000000,  0.000000,  0.128000],
                          [0.000000,  1.771000, -0.298000],
                          [1.533700, -0.885500, -0.298000],
                          [-1.533700, -0.885500, -0.298000]])
NH3_MASS = numpy.array([14.00307, 1.00783, 1.00783, 1.00783])
NH3_Z = numpy.array([7, 1, 1, 1])

CH4_COORDS = numpy.array([[0.000000,  0.000000,  0.000000],
                          [1.185000,  1.185000,  1.185000],
                          [-1.185000, -1.185000,  1.185000],
                          [-1.185000,  1.185000, -1.185000],
                          [1.185000, -1.185000, -1.185000]])
CH4_MASS = numpy.array([12.0, 1.00783, 1.00783, 1.00783, 1.00783])
CH4_Z = numpy.array([6, 1, 1, 1, 1])


class KnownValues(unittest.TestCase):

    # -----------------------------------------------------------------
    # the central cross-validation
    # -----------------------------------------------------------------

    def test_cross_validate_thermo(self):
        '''Independent validation of the whole mass-weighting / projection /
        diagonalisation chain against pyscf.hessian.thermo, which mass-weights
        in amu.  The amu -> m_e rescaling must cancel exactly against thermo's
        au2hz factor, so agreement is at machine precision, not merely 1e-4.
        '''
        model = HarmonicModel.from_mole(mol_h2o, hess_h2o, energy=e_h2o)
        ref = thermo.harmonic_analysis(mol_h2o, hess_h2o)

        self.assertEqual(model.nvib, 3)
        self.assertEqual(model.rotor_type, 'REGULAR')

        mine = numpy.sort(model.freq_wavenumber)
        theirs = numpy.sort(ref['freq_wavenumber'].real)
        rel = abs(mine - theirs) / abs(theirs)
        self.assertLess(rel.max(), 1e-4)          # the required tolerance
        self.assertLess(rel.max(), 1e-12)         # what is actually achieved

        # reduced masses (amu) must match too
        mu_mine = numpy.sort(model.reduced_mass)
        mu_ref = numpy.sort(ref['reduced_mass'])
        self.assertLess((abs(mu_mine - mu_ref) / mu_ref).max(), 1e-12)

        # and force constants: lambda_au = lambda_amu / AMU2AU
        fc_mine = numpy.sort(model.force_const)
        fc_ref = numpy.sort(ref['force_const_au']) / units.AMU2AU
        self.assertLess((abs(fc_mine - fc_ref) / abs(fc_ref)).max(), 1e-12)

        # Cartesian displacement vectors agree once thermo's amu normalisation
        # is undone (they differ only by the constant sqrt(AMU2AU)).
        cart = model.cartesian_modes * numpy.sqrt(units.AMU2AU)
        for k in range(model.nvib):
            self.assertLess(abs(abs(cart[k]) - abs(ref['norm_mode'][k])).max(), 1e-12)

        # sanity: the well-known STO-3G/RHF water frequencies
        self.assertAlmostEqual(model.freq_wavenumber[0], 2043.10691447, 5)
        self.assertAlmostEqual(model.freq_wavenumber[1], 4488.05268216, 5)
        self.assertAlmostEqual(model.freq_wavenumber[2], 4790.29531038, 5)

        # ZPE = half the sum of the frequencies
        self.assertAlmostEqual(model.zpe, .5 * model.freq.sum(), 14)
        self.assertAlmostEqual(model.zpe, 0.0257921721, 9)

    # -----------------------------------------------------------------
    # analytic reference
    # -----------------------------------------------------------------

    def test_analytic_diatomic(self):
        '''omega = sqrt(k/mu) exactly, with the masses entering through the
        public (amu) API so the amu -> electron-mass conversion is tested.
        '''
        for k, m1, m2 in ((0.5, 1.00783, 1.00783),      # H2-like
                          (0.37, 1.00783, 34.96885),    # HCl-like
                          (1.9, 12.0, 15.99491)):       # CO-like
            mass_amu = numpy.array([m1, m2])
            coords = numpy.array([[0., 0., 0.], [0., 0., 2.0]])
            hess = diatomic_stretch_hessian(k)
            model = HarmonicModel([1, 1], coords, mass_amu, hess)

            self.assertEqual(model.rotor_type, 'LINEAR')
            self.assertEqual(model.nvib, 1)

            mu_au = units.amu2au(m1 * m2 / (m1 + m2))
            omega_ref = numpy.sqrt(k / mu_au)
            self.assertLess(abs(model.freq[0] - omega_ref) / omega_ref, 1e-12)
            # `reduced_mass` is the Gaussian/thermo "effective mass along the
            # unit-normalised Cartesian displacement", NOT the bond reduced mass
            # mu = m1 m2/(m1+m2).  For a diatomic stretch it evaluates
            # analytically to m1 m2 (m1+m2) / (m1**2 + m2**2).
            mu_eff = m1 * m2 * (m1 + m2) / (m1**2 + m2**2)
            self.assertLess(abs(model.reduced_mass[0] - mu_eff) / mu_eff, 1e-12)

        # the same Hessian along an arbitrary bond direction
        rng = numpy.random.default_rng(8)
        n = rng.standard_normal(3)
        n /= numpy.linalg.norm(n)
        coords = numpy.array([[0., 0., 0.], 2.0 * n])
        model = HarmonicModel([1, 1], coords, [1.00783, 1.00783],
                              diatomic_stretch_hessian(0.5, n))
        mu_au = units.amu2au(1.00783 / 2)
        self.assertLess(abs(model.freq[0] - numpy.sqrt(0.5 / mu_au)) * numpy.sqrt(mu_au / 0.5),
                        1e-12)

    def test_mass_unit_keyword(self):
        coords = numpy.array([[0., 0., 0.], [0., 0., 2.0]])
        hess = diatomic_stretch_hessian(0.5)
        m_amu = HarmonicModel([1, 1], coords, [1.00783, 1.00783], hess)
        m_au = HarmonicModel([1, 1], coords, units.amu2au([1.00783, 1.00783]), hess,
                             mass_unit='au')
        self.assertAlmostEqual(abs(m_amu.freq - m_au.freq).max(), 0, 15)
        self.assertAlmostEqual(abs(m_amu.mass - m_au.mass).max(), 0, 12)
        # the stored mass is in electron masses, not amu
        self.assertAlmostEqual(m_amu.mass[0], 1.00783 * units.AMU2AU, 8)
        self.assertAlmostEqual(m_amu.mass_amu[0], 1.00783, 12)
        self.assertRaises(ValueError, HarmonicModel, [1, 1], coords,
                          [1., 1.], hess, None, 'kg')

    # -----------------------------------------------------------------
    # mode counting
    # -----------------------------------------------------------------

    def test_mode_counts(self):
        cases = [
            ('H2O', 101, numpy.array([8, 1, 1]), mol_h2o.atom_coords(),
             numpy.array([15.99491, 1.00783, 1.00783]), 'REGULAR', 3),
            ('NH3', 102, NH3_Z, NH3_COORDS, NH3_MASS, 'REGULAR', 6),
            ('CH4', 103, CH4_Z, CH4_COORDS, CH4_MASS, 'REGULAR', 9),
            ('CO2', 104, CO2_Z, CO2_COORDS, CO2_MASS, 'LINEAR', 4),
            ('HCN', 105, HCN_Z, HCN_COORDS, HCN_MASS, 'LINEAR', 4),
            ('H2', 106, numpy.array([1, 1]), numpy.array([[0., 0., 0.], [0., 0., 1.4]]),
             numpy.array([1.00783, 1.00783]), 'LINEAR', 1),
        ]
        for name, seed, z, coords, mass, rotor, nvib in cases:
            natm = len(z)
            model = HarmonicModel(z, coords, mass, psd_hessian(natm, seed),
                                  imaginary_policy='raise', verbose=0)
            self.assertEqual(model.rotor_type, rotor, name)
            self.assertEqual(model.nvib, nvib, name)
            self.assertEqual(model.freq.shape, (nvib,), name)
            self.assertEqual(model.modes.shape, (3 * natm, nvib), name)
            self.assertEqual(model.imaginary.shape, (nvib,), name)

    def test_single_atom(self):
        model = HarmonicModel([2], numpy.zeros((1, 3)), [4.0026], numpy.zeros((3, 3)))
        self.assertEqual(model.rotor_type, 'ATOM')
        self.assertEqual(model.nvib, 0)
        self.assertEqual(model.freq.shape, (0,))
        self.assertEqual(model.modes.shape, (3, 0))
        self.assertEqual(model.cartesian_modes.shape, (0, 1, 3))
        self.assertEqual(model.reduced_mass.shape, (0,))
        self.assertEqual(model.zpe, 0.0)
        self.assertEqual(int(model.imaginary.sum()), 0)
        model.dump_normal_modes(verbose=0)

    def test_projector_and_tr_vectors(self):
        mass_au = units.amu2au(NH3_MASS)
        tr = normal_modes.translation_rotation_vectors(mass_au, NH3_COORDS)
        self.assertEqual(tr.shape, (6, 12))
        # translations reproduce sqrt(m)
        self.assertAlmostEqual(abs(tr[0].reshape(4, 3)[:, 0] - mass_au**.5).max(), 0, 10)
        # rotations are orthogonal to translations for a COM-centred structure
        for i in range(3):
            for j in range(3, 6):
                self.assertAlmostEqual(tr[i].dot(tr[j]) / (numpy.linalg.norm(tr[i])
                                                           * numpy.linalg.norm(tr[j])), 0, 12)

        proj, basis = normal_modes.projector_trans_rot(mass_au, NH3_COORDS)
        self.assertEqual(basis.shape, (12, 6))
        self.assertAlmostEqual(abs(basis.T.dot(basis) - numpy.eye(6)).max(), 0, 12)
        self.assertAlmostEqual(abs(proj.dot(proj) - proj).max(), 0, 11)
        self.assertAlmostEqual(abs(proj.dot(basis) - basis).max(), 0, 11)
        # every TR vector is annihilated
        for row in tr:
            self.assertLess(numpy.linalg.norm(basis.T.dot(row)) / numpy.linalg.norm(row), 1e-12)

    def test_linear_rotation_vector_vanishes(self):
        mass_au = units.amu2au(CO2_MASS)
        tr = normal_modes.translation_rotation_vectors(mass_au, CO2_COORDS)
        # row 5 is the rotation about the axis with the smallest moment, which
        # for a linear molecule is the figure axis: it vanishes identically.
        self.assertAlmostEqual(numpy.linalg.norm(tr[5]), 0, 12)
        self.assertGreater(numpy.linalg.norm(tr[3]), 1.0)
        self.assertGreater(numpy.linalg.norm(tr[4]), 1.0)

    def test_projector_rejects_wrong_rotor_type(self):
        '''Forcing REGULAR on an exactly linear molecule makes the six TR
        vectors linearly dependent; that must be a loud RuntimeError, not a
        silently wrong mode count.'''
        mass_au = units.amu2au(CO2_MASS)
        self.assertRaises(RuntimeError, normal_modes.projector_trans_rot,
                          mass_au, CO2_COORDS, 'REGULAR')
        self.assertRaises(ValueError, normal_modes.projector_trans_rot,
                          mass_au, CO2_COORDS, 'NONSENSE')
        # 3N-6 < 0 for a diatomic misclassified as REGULAR
        self.assertRaises(RuntimeError, normal_modes.projector_trans_rot,
                          units.amu2au([1., 1.]), numpy.array([[0., 0., 0.], [0., 0., 1.4]]),
                          'REGULAR')

    # -----------------------------------------------------------------
    # invariance -- the crux
    # -----------------------------------------------------------------

    def test_invariance_translation(self):
        ref = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        coords = mol_h2o.atom_coords()
        mass = mol_h2o.atom_mass_list(isotope_avg=True)
        z = mol_h2o.atom_charges()
        for shift in ([1., 0., 0.], [-3.5, 2.25, 11.0]):
            moved = HarmonicModel(z, coords + numpy.array(shift), mass, hess_h2o)
            self.assertLess(abs(moved.freq - ref.freq).max() / abs(ref.freq).max(), 1e-14)
            # the stored geometry is COM-centred, so it is literally the same
            self.assertLess(abs(moved.coords - ref.coords).max(), 1e-12)
            self.assertLess(abs(abs(moved.modes) - abs(ref.modes)).max(), 1e-10)

    def test_invariance_rotation(self):
        '''Rotating the geometry *and* the Hessian consistently must leave the
        frequencies unchanged.'''
        rng = numpy.random.default_rng(2718)
        ref = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        coords = mol_h2o.atom_coords()
        mass = mol_h2o.atom_mass_list(isotope_avg=True)
        z = mol_h2o.atom_charges()
        h2 = normal_modes.reshape_hessian(hess_h2o, 3)

        worst = 0.
        for _ in range(20):
            rot = random_rotation(rng)
            rotated = HarmonicModel(z, coords.dot(rot), mass, rotate_hessian(h2, rot))
            err = abs(rotated.freq - ref.freq).max() / abs(ref.freq).max()
            worst = max(worst, err)
            # the modes rotate with the frame
            lref = ref.modes.reshape(3, 3, 3)
            lrot = rotated.modes.reshape(3, 3, 3)
            expect = numpy.einsum('adk,dc->ack', lref, rot)
            # up to the per-mode sign convention
            self.assertLess(abs(abs(expect) - abs(lrot)).max(), 1e-9)
        self.assertLess(worst, 1e-9)

    def test_invariance_permutation(self):
        '''Permuting the atoms (coords, masses, charges and Hessian blocks
        together) must give the same frequency spectrum.'''
        rng = numpy.random.default_rng(1414)
        ref = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        coords = mol_h2o.atom_coords()
        mass = mol_h2o.atom_mass_list(isotope_avg=True)
        z = mol_h2o.atom_charges()
        h4 = numpy.asarray(hess_h2o)

        for perm in ([1, 0, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]):
            perm = numpy.array(perm)
            model = HarmonicModel(z[perm], coords[perm], mass[perm],
                                  h4[numpy.ix_(perm, perm)])
            self.assertEqual(model.nvib, ref.nvib)
            self.assertLess(abs(numpy.sort(model.freq) - numpy.sort(ref.freq)).max()
                            / abs(ref.freq).max(), 1e-13)
            self.assertLess(abs(model.zpe - ref.zpe) / ref.zpe, 1e-13)

        # and a random permutation of a bigger system
        natm = 5
        hess = psd_hessian(natm, 77)
        base = HarmonicModel(CH4_Z, CH4_COORDS, CH4_MASS, hess)
        perm = rng.permutation(natm)
        h4b = hess.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
        model = HarmonicModel(CH4_Z[perm], CH4_COORDS[perm], CH4_MASS[perm],
                              h4b[numpy.ix_(perm, perm)])
        self.assertLess(abs(numpy.sort(model.freq) - numpy.sort(base.freq)).max()
                        / abs(base.freq).max(), 1e-12)

    def test_invariance_combined(self):
        '''Translation + rotation + permutation all at once.'''
        rng = numpy.random.default_rng(6161)
        ref = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        coords = mol_h2o.atom_coords()
        mass = mol_h2o.atom_mass_list(isotope_avg=True)
        z = mol_h2o.atom_charges()
        h2 = normal_modes.reshape_hessian(hess_h2o, 3)

        rot = random_rotation(rng)
        perm = numpy.array([2, 0, 1])
        new_h = rotate_hessian(h2, rot).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
        model = HarmonicModel(z[perm], coords.dot(rot)[perm] + numpy.array([4., -2., 9.]),
                              mass[perm], new_h[numpy.ix_(perm, perm)])
        self.assertLess(abs(numpy.sort(model.freq) - numpy.sort(ref.freq)).max()
                        / abs(ref.freq).max(), 1e-9)

    # -----------------------------------------------------------------
    # phase convention and orthonormality
    # -----------------------------------------------------------------

    def test_modes_orthonormal(self):
        for z, coords, mass, seed in ((NH3_Z, NH3_COORDS, NH3_MASS, 3),
                                      (CO2_Z, CO2_COORDS, CO2_MASS, 4),
                                      (CH4_Z, CH4_COORDS, CH4_MASS, 5)):
            model = HarmonicModel(z, coords, mass, psd_hessian(len(z), seed))
            gram = model.modes.T.dot(model.modes)
            self.assertAlmostEqual(abs(gram - numpy.eye(model.nvib)).max(), 0, 12)
            self.assertAlmostEqual(abs((model.modes**2).sum(axis=0) - 1).max(), 0, 12)
        model = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        self.assertAlmostEqual(abs(model.modes.T.dot(model.modes) - numpy.eye(3)).max(), 0, 13)

    def test_phase_convention_deterministic(self):
        a = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        b = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        self.assertAlmostEqual(abs(a.modes - b.modes).max(), 0, 15)
        # the pivot element (largest |.|, lowest index among ties) is positive
        for k in range(a.nvib):
            col = a.modes[:, k]
            amax = abs(col).max()
            idx = int(numpy.argmax(abs(col) >= amax * (1 - 1e-8)))
            self.assertGreater(col[idx], 0)

    def test_phase_convention_is_only_a_convention(self):
        '''Disabling the phase fix must change nothing but signs.'''
        fixed = HarmonicModel.from_mole(mol_h2o, hess_h2o, fix_phase=True)
        raw = HarmonicModel.from_mole(mol_h2o, hess_h2o, fix_phase=False)
        self.assertAlmostEqual(abs(fixed.freq - raw.freq).max(), 0, 15)
        self.assertAlmostEqual(abs(fixed.zpe - raw.zpe), 0, 15)
        self.assertAlmostEqual(abs(abs(fixed.modes) - abs(raw.modes)).max(), 0, 15)
        # each column differs by at most a sign
        for k in range(fixed.nvib):
            s = numpy.sign(fixed.modes[:, k].dot(raw.modes[:, k]))
            self.assertAlmostEqual(abs(fixed.modes[:, k] - s * raw.modes[:, k]).max(), 0, 15)
        # flipping an eigenvector by hand and re-fixing restores the convention
        flipped = raw.modes.copy()
        flipped[:, ::2] *= -1
        normal_modes.fix_mode_phase(flipped)
        self.assertAlmostEqual(abs(flipped - fixed.modes).max(), 0, 15)

    def test_fix_mode_phase_tie_breaking(self):
        '''With exactly-tied maxima the lowest index wins, deterministically.'''
        # column 0: |.| ties at indices 0,1,2 -> pivot is index 0, value -1,
        #           so the whole column is negated.
        # column 1: unique max |-2| at index 1 -> negated as well.
        m = numpy.array([[-1., 0.], [1., -2.], [1., 0.5]])
        normal_modes.fix_mode_phase(m)
        self.assertAlmostEqual(abs(m[:, 0] - numpy.array([1., -1., -1.])).max(), 0, 15)
        self.assertAlmostEqual(abs(m[:, 1] - numpy.array([0., 2., -0.5])).max(), 0, 15)

        # already-conforming columns are left alone
        m2 = numpy.array([[1., 0.], [-1., 2.], [-1., -0.5]])
        before = m2.copy()
        normal_modes.fix_mode_phase(m2)
        self.assertAlmostEqual(abs(m2 - before).max(), 0, 15)

        # a zero column is untouched
        z = numpy.zeros((3, 1))
        normal_modes.fix_mode_phase(z)
        self.assertAlmostEqual(abs(z).max(), 0, 15)
        self.assertRaises(ValueError, normal_modes.fix_mode_phase, numpy.zeros(3))

    # -----------------------------------------------------------------
    # isotopes
    # -----------------------------------------------------------------

    def test_isotope_d2o(self):
        h2o = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        d2o = HarmonicModel.from_mole(mol_h2o, hess_h2o,
                                      mass=[15.99491, 2.01410, 2.01410])
        self.assertEqual(d2o.nvib, h2o.nvib)
        self.assertEqual(d2o.rotor_type, h2o.rotor_type)
        ratio = h2o.freq / d2o.freq
        # sqrt(m_D/m_H) = 1.374 for a pure X-H stretch; the O recoil moves each
        # mode a little off that value.
        for r in ratio:
            self.assertGreater(r, 1.30)
            self.assertLess(r, 1.42)
        self.assertAlmostEqual(ratio[-1], 1.36526, 4)   # antisym. OH stretch
        self.assertAlmostEqual(ratio[1], 1.38616, 4)    # sym. OH stretch
        self.assertAlmostEqual(ratio[0], 1.36711, 4)    # bend
        # heavier -> lower ZPE
        self.assertLess(d2o.zpe, h2o.zpe)
        # reduced masses go up
        self.assertTrue(numpy.all(d2o.reduced_mass > h2o.reduced_mass))
        # the Hessian is untouched by isotopic substitution
        self.assertAlmostEqual(abs(d2o.hessian - h2o.hessian).max(), 0, 15)

    def test_isotope_avg_flag(self):
        avg = HarmonicModel.from_mole(mol_h2o, hess_h2o, isotope_avg=True)
        pure = HarmonicModel.from_mole(mol_h2o, hess_h2o, isotope_avg=False)
        # isotope_avg=False gives the integer mass numbers 16 and 1
        self.assertAlmostEqual(abs(pure.mass_amu - numpy.array([16., 1., 1.])).max(), 0, 10)
        self.assertAlmostEqual(abs(avg.mass_amu - numpy.array([15.999, 1.008, 1.008])).max(),
                               0, 3)
        rel = abs(avg.freq - pure.freq).max() / abs(avg.freq).max()
        self.assertGreater(rel, 1e-4)      # the flag really does something
        self.assertLess(rel, 1e-2)         # but only at the 0.4% level (measured 0.0037)

    # -----------------------------------------------------------------
    # imaginary frequencies
    # -----------------------------------------------------------------

    def _h2o_with_one_imaginary(self):
        '''Flip the sign of the lowest eigenvalue of the water Hessian in the
        vibrational subspace, leaving the other two untouched.'''
        ref = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        sqm = numpy.repeat(ref.mass**.5, 3)
        l0 = ref.modes[:, 0]
        lam0 = ref.force_const[0]
        dh_mw = -2 * lam0 * numpy.outer(l0, l0)
        dh = sqm[:, None] * dh_mw * sqm[None, :]
        return ref, ref.hessian + dh

    def test_imaginary_policy_raise_is_default(self):
        ref, hess = self._h2o_with_one_imaginary()
        z, coords, mass = (mol_h2o.atom_charges(), mol_h2o.atom_coords(),
                           mol_h2o.atom_mass_list(isotope_avg=True))
        with self.assertRaises(RuntimeError) as ctx:
            HarmonicModel(z, coords, mass, hess)
        msg = str(ctx.exception)
        self.assertIn('imaginary', msg)
        self.assertIn('2043', msg)                    # names the offending wavenumber

    def test_imaginary_policy_warn_and_ignore(self):
        ref, hess = self._h2o_with_one_imaginary()
        z, coords, mass = (mol_h2o.atom_charges(), mol_h2o.atom_coords(),
                           mol_h2o.atom_mass_list(isotope_avg=True))
        for policy in ('warn', 'ignore'):
            out = StringIO()
            model = HarmonicModel(z, coords, mass, hess, imaginary_policy=policy,
                                  stdout=out, verbose=lib.logger.WARN)
            # nothing is dropped
            self.assertEqual(model.nvib, 3)
            self.assertEqual(model.freq.shape, (3,))
            self.assertEqual(model.modes.shape, (9, 3))
            self.assertEqual(int(model.imaginary.sum()), 1)
            self.assertTrue(model.imaginary[0])
            self.assertFalse(model.imaginary[1:].any())
            # the imaginary mode sorts first and is stored as a negative number
            self.assertLess(model.freq[0], 0)
            self.assertAlmostEqual(model.freq[0], -ref.freq[0], 12)
            # the other two are unchanged
            self.assertLess(abs(model.freq[1:] - ref.freq[1:]).max() / ref.freq[-1], 1e-12)
            # the ZPE excludes it
            self.assertAlmostEqual(model.zpe, .5 * ref.freq[1:].sum(), 14)
            if policy == 'warn':
                self.assertIn('imaginary', out.getvalue())
            else:
                self.assertNotIn('imaginary', out.getvalue())

    def test_imaginary_policy_validation(self):
        z, coords, mass = (mol_h2o.atom_charges(), mol_h2o.atom_coords(),
                           mol_h2o.atom_mass_list(isotope_avg=True))
        self.assertRaises(ValueError, HarmonicModel, z, coords, mass, hess_h2o,
                          None, 'amu', 'shout')

    def test_near_zero_frequency_warning(self):
        '''A retained mode below FREQ_ZERO_TOL must be flagged.'''
        ref = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        sqm = numpy.repeat(ref.mass**.5, 3)
        l0 = ref.modes[:, 0]
        target = 0.1 * normal_modes.FREQ_ZERO_TOL          # 1 cm^-1
        dh_mw = (target**2 - ref.force_const[0]) * numpy.outer(l0, l0)
        hess = ref.hessian + sqm[:, None] * dh_mw * sqm[None, :]
        out = StringIO()
        model = HarmonicModel(mol_h2o.atom_charges(), mol_h2o.atom_coords(),
                              mol_h2o.atom_mass_list(isotope_avg=True), hess,
                              stdout=out, verbose=lib.logger.WARN)
        self.assertAlmostEqual(model.freq[0], target, 12)
        self.assertIn('below', out.getvalue())
        # and no warning for the untouched molecule
        out = StringIO()
        HarmonicModel.from_mole(mol_h2o, hess_h2o, stdout=out, verbose=lib.logger.WARN)
        self.assertEqual(out.getvalue(), '')

    # -----------------------------------------------------------------
    # robustness
    # -----------------------------------------------------------------

    def test_asymmetric_hessian_tolerated(self):
        rng = numpy.random.default_rng(555)
        ref = HarmonicModel.from_mole(mol_h2o, hess_h2o)
        h2 = normal_modes.reshape_hessian(hess_h2o, 3)
        b = rng.standard_normal((9, 9))
        anti = b - b.T
        for scale in (1e-9, 1e-6, 1e-3):
            out = StringIO()
            model = HarmonicModel(mol_h2o.atom_charges(), mol_h2o.atom_coords(),
                                  mol_h2o.atom_mass_list(isotope_avg=True),
                                  h2 + scale * anti, stdout=out, verbose=lib.logger.WARN)
            # (H + A + (H + A)^T)/2 == H exactly for antisymmetric A, so the
            # frequencies are bit-for-bit unchanged.
            self.assertLess(abs(model.freq - ref.freq).max() / abs(ref.freq).max(), 1e-14)
            self.assertAlmostEqual(model.hessian_asymmetry,
                                   2 * scale * abs(anti).max(), 12)
            if scale * 2 * abs(anti).max() > normal_modes.HESSIAN_ASYMMETRY_TOL:
                self.assertIn('asymmetric', out.getvalue())
            else:
                self.assertEqual(out.getvalue(), '')
        self.assertLess(ref.hessian_asymmetry, 1e-12)

    def test_hessian_shape_handling(self):
        h4 = numpy.asarray(hess_h2o)
        self.assertEqual(h4.shape, (3, 3, 3, 3))
        h2 = normal_modes.reshape_hessian(h4, 3)
        self.assertEqual(h2.shape, (9, 9))
        a = HarmonicModel.from_mole(mol_h2o, h4)
        b = HarmonicModel.from_mole(mol_h2o, h2)
        self.assertAlmostEqual(abs(a.freq - b.freq).max(), 0, 15)
        self.assertAlmostEqual(abs(a.modes - b.modes).max(), 0, 15)
        self.assertAlmostEqual(abs(a.hessian - b.hessian).max(), 0, 15)
        # the flattening convention matches PySCF's: H[a,b,x,y] -> H[3a+x, 3b+y]
        for a_ in range(3):
            for b_ in range(3):
                self.assertAlmostEqual(
                    abs(h2[3 * a_:3 * a_ + 3, 3 * b_:3 * b_ + 3] - h4[a_, b_]).max(), 0, 15)

    def test_bad_input(self):
        z, coords, mass = (mol_h2o.atom_charges(), mol_h2o.atom_coords(),
                           mol_h2o.atom_mass_list(isotope_avg=True))
        # mismatched natm
        self.assertRaises(ValueError, HarmonicModel, z[:2], coords, mass, hess_h2o)
        self.assertRaises(ValueError, HarmonicModel, z, coords[:2], mass, hess_h2o)
        self.assertRaises(ValueError, HarmonicModel, z, coords, mass[:2], hess_h2o)
        # negative / zero mass
        bad_mass = mass.copy()
        bad_mass[1] = -1.0
        self.assertRaises(ValueError, HarmonicModel, z, coords, bad_mass, hess_h2o)
        bad_mass[1] = 0.0
        self.assertRaises(ValueError, HarmonicModel, z, coords, bad_mass, hess_h2o)
        # NaN / inf in the Hessian
        bad_hess = numpy.array(hess_h2o)
        bad_hess[0, 0, 0, 0] = numpy.nan
        self.assertRaises(ValueError, HarmonicModel, z, coords, mass, bad_hess)
        bad_hess[0, 0, 0, 0] = numpy.inf
        self.assertRaises(ValueError, HarmonicModel, z, coords, mass, bad_hess)
        # NaN in the coordinates
        bad_coords = coords.copy()
        bad_coords[2, 1] = numpy.nan
        self.assertRaises(ValueError, HarmonicModel, z, bad_coords, mass, hess_h2o)
        # wrong Hessian shape
        self.assertRaises(ValueError, HarmonicModel, z, coords, mass, numpy.zeros((8, 8)))
        self.assertRaises(ValueError, HarmonicModel, z, coords, mass, numpy.zeros((3, 3, 3)))
        self.assertRaises(ValueError, HarmonicModel, z, coords, mass, numpy.zeros((2, 2, 3, 3)))

    def test_attributes_and_dump(self):
        model = HarmonicModel.from_mole(mol_h2o, hess_h2o, energy=e_h2o)
        self.assertEqual(model.natm, 3)
        self.assertEqual(model.n3, 9)
        self.assertIs(model.mol, mol_h2o)
        self.assertAlmostEqual(model.energy, e_h2o, 12)
        self.assertEqual(model.hessian.shape, (9, 9))
        self.assertAlmostEqual(abs(model.hessian - model.hessian.T).max(), 0, 15)
        # geometry is stored centred at the centre of mass
        com = alignment.center_of_mass(model.mass, model.coords)
        self.assertAlmostEqual(abs(com).max(), 0, 13)
        self.assertEqual(model.cartesian_modes.shape, (3, 3, 3))
        out = StringIO()
        model.stdout = out
        model.dump_normal_modes(verbose=lib.logger.NOTE)
        text = out.getvalue()
        self.assertIn('REGULAR', text)
        self.assertIn('ZPE', text)
        self.assertIn('4790', text)
        # build() is idempotent
        f0 = model.freq.copy()
        model.build()
        self.assertAlmostEqual(abs(model.freq - f0).max(), 0, 15)
        self.assertIs(model.kernel.__func__, HarmonicModel.build)

    def test_low_level_helpers(self):
        mass_au = units.amu2au(NH3_MASS)
        h = psd_hessian(4, 12)
        hmw = normal_modes.mass_weighted_hessian(mass_au, h)
        inv = numpy.repeat(mass_au**-.5, 3)
        self.assertAlmostEqual(abs(hmw - h * inv[:, None] * inv[None, :]).max(), 0, 14)
        self.assertRaises(ValueError, normal_modes.mass_weighted_hessian,
                          mass_au, numpy.zeros((9, 9)))
        self.assertRaises(ValueError, normal_modes.mass_weighted_hessian,
                          -mass_au, h)

        _, basis = normal_modes.projector_trans_rot(mass_au, NH3_COORDS)
        hp = normal_modes.project_hessian(hmw, basis)
        self.assertEqual(hp.shape, (6, 6))
        self.assertAlmostEqual(abs(hp - basis.T.dot(hmw).dot(basis)).max(), 0, 13)
        self.assertRaises(ValueError, normal_modes.project_hessian, hmw, basis[:5])

        res = normal_modes.harmonic_analysis(mass_au, NH3_COORDS, h)
        self.assertEqual(res['nvib'], 6)
        self.assertEqual(res['rotor_type'], 'REGULAR')
        self.assertTrue(numpy.all(numpy.diff(res['freq']) >= 0))
        self.assertAlmostEqual(
            abs(numpy.sort(res['force_const']) - numpy.sort(numpy.linalg.eigvalsh(hp))).max(),
            0, 13)

    def test_sorting_is_ascending_and_stable(self):
        model = HarmonicModel(CH4_Z, CH4_COORDS, CH4_MASS, psd_hessian(5, 909))
        self.assertTrue(numpy.all(numpy.diff(model.freq) >= 0))
        # a degenerate spectrum keeps the eigensolver's relative order
        mass = numpy.array([1.00783, 1.00783])
        coords = numpy.array([[0., 0., 0.], [0., 0., 2.0]])
        m = HarmonicModel([1, 1], coords, mass, diatomic_stretch_hessian(0.5))
        self.assertEqual(m.freq.shape, (1,))


if __name__ == '__main__':
    print('Full Tests for pyscf.vibronic.normal_modes')
    unittest.main()
