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

'''End-to-end tests of the high-level :class:`pyscf.vibronic.FranckCondon`
workflow.

These exercise the whole chain -- alignment, normal modes, Duschinsky
transformation, overlaps, energies and spectra -- rather than any single
component.  The load-bearing test is
:meth:`KnownValues.test_displaced_diatomic_is_poisson`, which drives the *public*
API end to end and compares against the analytic Poisson progression, so a bug
anywhere in the chain (mass weighting, Eckart alignment, the sign of ``K``, the
overlap recursion, the energy bookkeeping) shows up as a numerical failure.
'''

import math
import unittest
import numpy

from pyscf import gto, scf, lib
from pyscf import vibronic
from pyscf.vibronic import units
from pyscf.vibronic.normal_modes import HarmonicModel
from pyscf.vibronic.spectrum import trapezoid


def diatomic_model(bond, force_const, mass=(1.00794, 1.00794), energy=0.0):
    '''A synthetic 1-mode harmonic diatomic aligned along z.

    ``bond`` in bohr, ``force_const`` in Eh/bohr^2, ``mass`` in amu.  The Hessian
    is the exact harmonic Hessian of a stretch along z, so the analytic
    frequency ``sqrt(k/mu)`` is available for comparison.
    '''
    coords = numpy.array([[0., 0., -0.5 * bond], [0., 0., 0.5 * bond]])
    h = numpy.zeros((2, 2, 3, 3))
    h[0, 0, 2, 2] = h[1, 1, 2, 2] = force_const
    h[0, 1, 2, 2] = h[1, 0, 2, 2] = -force_const
    return HarmonicModel([1, 1], coords, list(mass), h, energy=energy,
                         verbose=0)


def rotate(coords, hess_4d, rot):
    '''Rotate a geometry and its (natm,natm,3,3) Hessian consistently.'''
    new_h = numpy.einsum('abxy,xc,yd->abcd', hess_4d, rot, rot)
    return coords.dot(rot), new_h


def random_rotation(seed):
    rng = numpy.random.RandomState(seed)
    q, _ = numpy.linalg.qr(rng.randn(3, 3))
    if numpy.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def water(basis='sto-3g'):
    return gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                 basis=basis, verbose=0)


class KnownValues(unittest.TestCase):

    # ---------------------------------------------------------------- trivial

    def test_identical_states_give_only_the_origin_line(self):
        '''Identical electronic states: J = I, K = 0, so only 0-0 survives.'''
        m = diatomic_model(1.4, 0.35, energy=-1.0)
        fc = vibronic.FranckCondon(m, None, m, None)
        res = fc.run(max_quanta=6)
        self.assertEqual(res.nstate, 7)
        origin = numpy.where(res.states.sum(axis=1) == 0)[0][0]
        self.assertAlmostEqual(res.fcf[origin], 1.0, 12)
        others = numpy.delete(res.fcf, origin)
        self.assertAlmostEqual(abs(others).max(), 0.0, 12)
        self.assertAlmostEqual(res.sum_rule, 1.0, 12)
        # E_00 = 0 because both the electronic energies and the ZPEs match
        self.assertAlmostEqual(res.e_adiabatic, 0.0, 14)
        self.assertAlmostEqual(res.e_00, 0.0, 14)

    # ------------------------------------------------- the load-bearing test

    def test_displaced_diatomic_is_poisson(self):
        '''Equal frequencies + pure displacement -> |<n|0>|^2 = e^-S S^n/n!.

        Driven entirely through the public API, so this validates the whole
        chain against an analytic result.  S is derived independently here from
        the reduced mass and the bond-length change, never from the code under
        test.
        '''
        k = 0.30
        m_amu = 12.0
        b_i, b_f = 2.20, 2.62
        mi = diatomic_model(b_i, k, mass=(m_amu, m_amu), energy=-10.0)
        mf = diatomic_model(b_f, k, mass=(m_amu, m_amu), energy=-9.7)

        # independent S: the mass-weighted displacement of a symmetric-stretch
        # normal coordinate for a homonuclear diatomic is sqrt(mu) * Delta b
        # with mu = m/2 the reduced mass (electron masses).
        mu = 0.5 * m_amu * units.AMU2AU
        omega = math.sqrt(k / mu)
        K_exact = math.sqrt(mu) * (b_i - b_f)
        S_exact = 0.5 * omega * K_exact ** 2

        fc = vibronic.FranckCondon(mi, None, mf, None)
        res = fc.run(max_quanta=30)

        dus = fc.duschinsky
        self.assertAlmostEqual(abs(dus.J[0, 0]), 1.0, 12)
        self.assertAlmostEqual(abs(dus.K[0]), abs(K_exact), 10)
        self.assertAlmostEqual(float(dus.huang_rhys[0]), S_exact, 10)
        self.assertAlmostEqual(float(res.freq_f[0]), omega, 12)

        # Poisson progression, through the public result object
        order = numpy.argsort(res.states[:, 0])
        for n, idx in enumerate(order):
            expect = math.exp(-S_exact) * S_exact ** n / math.factorial(n)
            self.assertAlmostEqual(res.fcf[idx], expect, 12)
        self.assertAlmostEqual(res.sum_rule, 1.0, 10)

        # energy bookkeeping, independently reconstructed
        e00 = (mf.energy - mi.energy) + mf.zpe - mi.zpe
        self.assertAlmostEqual(res.e_00, e00, 12)
        for n, idx in enumerate(order):
            self.assertAlmostEqual(res.energies[idx], e00 + n * omega, 12)

    def test_huang_rhys_sign_independent_of_displacement_direction(self):
        '''S depends on |K|, so compressing and stretching by the same amount
        must give the same progression.'''
        k, m_amu, b = 0.30, 12.0, 2.40
        d = 0.35
        out = []
        for b_f in (b + d, b - d):
            mi = diatomic_model(b, k, mass=(m_amu, m_amu))
            mf = diatomic_model(b_f, k, mass=(m_amu, m_amu))
            res = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=20)
            out.append(res.fcf[numpy.argsort(res.states[:, 0])])
        self.assertAlmostEqual(abs(out[0] - out[1]).max(), 0.0, 12)

    # -------------------------------------------------------- energy identities

    def test_energy_conventions(self):
        '''e_00 = e_adiabatic + zpe_f - zpe_i, and lines sit at
        e_00 + sum omega_f v_f.  Also checks that the three distinct energies
        (adiabatic, 0-0, individual vibronic) are not conflated.'''
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.22, mass=(12.0, 12.0), energy=-19.5)
        res = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=8)

        self.assertAlmostEqual(res.e_adiabatic, 0.5, 12)
        self.assertAlmostEqual(res.e_00, 0.5 + mf.zpe - mi.zpe, 12)
        # frequency drops, so ZPE drops, so E_00 < E_adiabatic here
        self.assertLess(res.e_00, res.e_adiabatic)
        for idx in range(res.nstate):
            n = int(res.states[idx, 0])
            self.assertAlmostEqual(res.energies[idx],
                                   res.e_00 + n * float(res.freq_f[0]), 12)

    def test_summary_units_round_trip(self):
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.22, mass=(12.0, 12.0), energy=-19.5)
        fc = vibronic.FranckCondon(mi, None, mf, None)
        fc.run(max_quanta=6)
        ev = fc.summary(unit='eV')
        au = fc.summary(unit='au')
        self.assertAlmostEqual(ev['e_00'], units.au2ev(au['e_00']), 10)
        cm = fc.summary(unit='cm-1')
        self.assertAlmostEqual(cm['e_00'], units.au2wavenumber(au['e_00']), 6)

    # ------------------------------------------------------------- invariance

    def test_spectrum_invariant_to_rigid_motion_of_the_final_state(self):
        '''Rotating and translating the final state (with its Hessian rotated
        consistently) must leave every Franck-Condon factor and every transition
        energy unchanged.  This is the test that proves L_f is rotated together
        with the geometry during Eckart alignment.'''
        k_i, k_f, m_amu = 0.30, 0.24, 12.0
        mi = diatomic_model(2.2, k_i, mass=(m_amu, m_amu), energy=-20.0)

        coords_f = numpy.array([[0., 0., -1.3], [0., 0., 1.3]])
        h_f = numpy.zeros((2, 2, 3, 3))
        h_f[0, 0, 2, 2] = h_f[1, 1, 2, 2] = k_f
        h_f[0, 1, 2, 2] = h_f[1, 0, 2, 2] = -k_f

        ref = None
        for seed in (0, 1, 2, 3):
            rot = random_rotation(seed)
            shift = numpy.array([0.7, -1.9, 3.1]) * (seed + 1)
            c_rot, h_rot = rotate(coords_f, h_f, rot)
            mf = HarmonicModel([1, 1], c_rot + shift, [m_amu, m_amu], h_rot,
                               energy=-19.5, verbose=0)
            res = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=12)
            order = numpy.argsort(res.states[:, 0])
            got = numpy.concatenate([res.fcf[order], res.energies[order]])
            if ref is None:
                ref = got
            else:
                self.assertAlmostEqual(abs(got - ref).max(), 0.0, 10)

    def test_water_spectrum_invariant_to_rigid_motion(self):
        '''The same invariance for a genuine polyatomic with real mode mixing.'''
        mol = water()
        mf_scf = scf.RHF(mol).run()
        hess = mf_scf.Hessian().kernel()
        mass = mol.atom_mass_list(isotope_avg=True)
        charges = mol.atom_charges()
        coords = mol.atom_coords()

        # a distinct "final" state: soften the Hessian and displace the geometry
        hess_f = 0.82 * hess
        disp = numpy.array([[0., 0., 0.07], [0., 0.05, -0.03], [0., -0.05, -0.03]])
        model_i = HarmonicModel(charges, coords, mass, hess, energy=-75.0, verbose=0)

        ref = None
        for seed in (0, 5, 11):
            rot = random_rotation(seed)
            c_rot, h_rot = rotate(coords + disp, hess_f, rot)
            model_f = HarmonicModel(charges, c_rot + numpy.array([2., -1., 4.]),
                                    mass, h_rot, energy=-74.7, verbose=0)
            res = vibronic.FranckCondon(model_i, None, model_f, None).run(
                max_quanta=4)
            got = numpy.concatenate([res.fcf, res.energies])
            if ref is None:
                ref = got
                self.assertGreater(res.sum_rule, 0.99)
            else:
                self.assertAlmostEqual(abs(got - ref).max(), 0.0, 10)

    # ------------------------------------------------------------ real molecule

    def test_water_workflow_diagnostics(self):
        '''Real polyatomic: a symmetry-breaking distortion must produce genuine
        Duschinsky mode mixing, and all diagnostics must be within bounds.

        The distortion is deliberately *asymmetric* (only one O-H bond is
        lengthened).  A symmetric distortion preserves C2v, so the modes cannot
        mix by symmetry and ``max_offdiag_J`` stays at the 1e-15 level -- which
        would make a "mode mixing is present" assertion vacuous.

        The final-state Hessian is recomputed by a real SCF at the distorted
        geometry.  Simply *scaling* the initial Hessian would not do: a scaled
        matrix has exactly the same eigenvectors, so J would be the identity by
        construction and the test would prove nothing about Duschinsky mixing.
        '''
        mol = water()
        mf_scf = scf.RHF(mol).run()
        hess = mf_scf.Hessian().kernel()
        mass = mol.atom_mass_list(isotope_avg=True)
        model_i = HarmonicModel(mol.atom_charges(), mol.atom_coords(), mass, hess,
                                energy=mf_scf.e_tot, verbose=0)

        # final state: one O-H stretched (breaks C2v), Hessian recomputed there
        mol_f = gto.M(atom='O 0 0 0.1173; H 0 0.8329 -0.5161; H 0 -0.7572 -0.4692',
                      basis='sto-3g', verbose=0)
        mf_f = scf.RHF(mol_f).run()
        hess_f = mf_f.Hessian().kernel()
        model_f = HarmonicModel(mol_f.atom_charges(), mol_f.atom_coords(), mass,
                                hess_f, energy=mf_scf.e_tot + 0.25, verbose=0,
                                imaginary_policy='warn')

        # the distorted geometry is not a stationary point, so tolerate a small
        # imaginary mode rather than pretending it is a minimum
        fc = vibronic.FranckCondon(model_i, None, model_f, None)
        res = fc.run(max_quanta=6, allow_imaginary=True)
        dus = fc.duschinsky

        self.assertEqual(model_i.nvib, 3)
        self.assertEqual(model_f.nvib, 3)
        # genuine mode mixing: this is not a diagonal-J special case
        self.assertGreater(dus.diagnostics['max_offdiag_J'], 1e-2)
        # the displacement lies almost entirely inside the vibrational subspace
        self.assertLess(dus.diagnostics['excluded_mode_norm'], 1e-3)
        self.assertGreater(res.sum_rule, 0.95)
        self.assertAlmostEqual(res.e_adiabatic, 0.25, 10)

    def test_j_orthogonality_error_vanishes_with_the_geometry_change(self):
        '''J = L_f^T L_i is exactly orthogonal only when both states span the
        *same* vibrational subspace.

        Two different geometries have different rotational subspaces, so the
        vibrational subspaces differ slightly and J^T J deviates from the
        identity.  Asserting a fixed small tolerance would therefore be wrong;
        the meaningful statements are that the deviation (a) is exactly zero at
        zero displacement and (b) shrinks monotonically as the distortion
        shrinks.  Measured for water/STO-3G: 1.8e-15 at 0%, 8.8e-5 at 2%,
        5.3e-4 at 5%, 2.0e-3 at 10%, 7.0e-3 at 20%.
        '''
        mol = water()
        mf_scf = scf.RHF(mol).run()
        hess = mf_scf.Hessian().kernel()
        mass = mol.atom_mass_list(isotope_avg=True)
        charges = mol.atom_charges()
        c0 = mol.atom_coords()
        model_i = HarmonicModel(charges, c0, mass, hess, energy=-75.0, verbose=0)

        errs = []
        for scale in (1.00, 1.02, 1.05, 1.10, 1.20):
            c_f = c0.copy()
            c_f[1, 1] *= scale
            c_f[2, 1] *= scale
            model_f = HarmonicModel(charges, c_f, mass, 0.80 * hess,
                                    energy=-74.75, verbose=0)
            dus = vibronic.duschinsky_transform(model_i, model_f, verbose=0)
            direct = abs(dus.J.T.dot(dus.J) - numpy.eye(3)).max()
            self.assertAlmostEqual(direct, dus.diagnostics['orthogonality_error'], 12)
            errs.append(direct)

        # identical geometries -> exactly orthogonal
        self.assertLess(errs[0], 1e-13)
        # and monotonically increasing with the distortion
        for a, b in zip(errs, errs[1:]):
            self.assertLess(a, b)
        # even a 20% distortion stays a small perturbation
        self.assertLess(errs[-1], 1e-2)

    # ------------------------------------------------------- absorption/emission

    def test_absorption_and_emission_are_mirror_images(self):
        '''Swapping which state is "initial" turns the absorption problem into
        the emission problem; the 0-0 photon energies must coincide and the
        vibronic progressions must run in opposite directions from it.'''
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.26, mass=(12.0, 12.0), energy=-19.5)

        absorp = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=10)
        emiss = vibronic.FranckCondon(mf, None, mi, None).run(max_quanta=10)

        a = absorp.stick_spectrum(kind='absorption')
        e = emiss.stick_spectrum(kind='emission')
        self.assertTrue(numpy.all(a.energies > 0))
        self.assertTrue(numpy.all(e.energies > 0))
        # both origins sit at |E_00| of their own direction; the two E_00 values
        # differ only by the sign convention
        self.assertAlmostEqual(absorp.e_00, -emiss.e_00, 12)
        # absorption progresses to higher photon energy, emission to lower
        self.assertGreater(a.energies.max(), a.energies.min())
        self.assertLess(e.energies.min(), -emiss.e_00 + 1e-12)

    def test_emission_with_wrong_state_order_raises(self):
        '''Asking for emission when the initial state is the lower one must
        raise, not silently produce negative photon energies.'''
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.26, mass=(12.0, 12.0), energy=-19.5)
        res = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=4)
        self.assertRaises(ValueError, res.stick_spectrum, kind='emission')

    def test_broadening_conserves_area(self):
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.26, mass=(12.0, 12.0), energy=-19.5)
        res = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=20)
        sticks = res.stick_spectrum(kind='absorption')
        # StickSpectrum.broaden returns a BroadenedSpectrum object; the
        # module-level spectrum.broaden() function returns a (grid, signal) tuple.
        # `padding` is an energy in `unit`, not a multiple of the width; the
        # default 5*width is what keeps the edge lineshapes inside the grid.
        spec = sticks.broaden(profile='gaussian', width=200, unit='cm-1',
                              npoints=40001)
        area = trapezoid(spec.y, spec.x)
        self.assertAlmostEqual(area / sticks.intensities.sum(), 1.0, 6)

        # the low-level function takes stick positions in Hartree and must agree
        grid, signal = vibronic.broaden(
            sticks.energies, sticks.intensities,
            profile='gaussian', width=200, unit='cm-1', npoints=40001)
        self.assertAlmostEqual(abs(signal - spec.y).max(), 0.0, 12)
        self.assertAlmostEqual(abs(grid - spec.x).max(), 0.0, 10)

        # a grid padded far too narrowly must lose area -- documents that the
        # area guarantee is conditional on the grid covering the lineshapes
        narrow = sticks.broaden(profile='gaussian', width=200, unit='cm-1',
                                npoints=40001, padding=12.0)
        self.assertLess(trapezoid(narrow.y, narrow.x) / sticks.intensities.sum(),
                        0.99)

    # ------------------------------------------------------------- temperature

    def test_zero_temperature_limit_of_the_finite_temperature_path(self):
        '''A very low temperature must reproduce the T = 0 result.'''
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.26, mass=(12.0, 12.0), energy=-19.5)
        cold = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=12)
        warm = vibronic.FranckCondon(mi, None, mf, None).run(
            max_quanta=12, temperature=1.0)
        # at 1 K only |0_i> is populated, so the surviving lines must match
        key = {tuple(s): f for s, f in zip(cold.states, cold.fcf)}
        for s_f, s_i, f in zip(warm.states, warm.init_states, warm.fcf):
            if s_i.sum() == 0:
                self.assertAlmostEqual(f, key[tuple(s_f)], 12)

    def test_finite_temperature_adds_hot_bands(self):
        '''At a temperature comparable to the vibrational quantum, transitions
        from excited initial states must appear and carry intensity below the
        0-0 energy.'''
        mi = diatomic_model(2.2, 0.045, mass=(30.0, 30.0), energy=-20.0)
        mf = diatomic_model(2.4, 0.040, mass=(30.0, 30.0), energy=-19.5)
        fc = vibronic.FranckCondon(mi, None, mf, None)
        res = fc.run(max_quanta=8, temperature=600.0)
        hot = res.init_states.sum(axis=1) > 0
        self.assertTrue(hot.any())
        weight = res.populations * res.fcf
        self.assertGreater(weight[hot].sum(), 1e-4)
        # some hot lines sit below E_00
        self.assertLess(res.energies[hot].min(), res.e_00)

    # -------------------------------------------------------------- isotopes

    def test_with_isotopes_changes_frequencies_consistently(self):
        mol = water()
        mf_scf = scf.RHF(mol).run()
        hess = mf_scf.Hessian().kernel()
        fc = vibronic.FranckCondon(mol, hess, mol, hess,
                                   initial_energy=mf_scf.e_tot,
                                   final_energy=mf_scf.e_tot + 0.2)
        fc.run(max_quanta=2)
        light = fc.model_i.freq_wavenumber.copy()

        d2o = fc.with_isotopes([15.9949, 2.0141, 2.0141])
        d2o.run(max_quanta=2)
        heavy = d2o.model_i.freq_wavenumber
        self.assertEqual(len(light), len(heavy))
        # every mode softens on deuteration
        self.assertTrue(numpy.all(heavy < light))
        # the O-H stretches shift by roughly sqrt(2)
        ratio = light[-1] / heavy[-1]
        self.assertTrue(1.28 < ratio < 1.42, 'ratio = %r' % ratio)

    def test_with_isotopes_rejects_prebuilt_models(self):
        m = diatomic_model(2.2, 0.3)
        fc = vibronic.FranckCondon(m, None, m, None)
        self.assertRaises(ValueError, fc.with_isotopes, [2.0141, 2.0141])

    # ------------------------------------------------------------- truncation

    def test_truncation_is_reported_not_silent(self):
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.9, 0.26, mass=(12.0, 12.0), energy=-19.5)
        res = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=2)
        self.assertIn('n_enumerated', res.truncation)
        # a large displacement truncated at 2 quanta cannot satisfy the sum rule
        self.assertLess(res.sum_rule, 0.999)
        report = vibronic.analysis.sum_rule_report(res)
        self.assertAlmostEqual(report['missing'], 1.0 - res.sum_rule, 12)

        converged = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=40)
        self.assertGreater(converged.sum_rule, res.sum_rule)
        self.assertAlmostEqual(converged.sum_rule, 1.0, 8)

    def test_closure_rule_target_is_one_over_det_j(self):
        r'''The Franck-Condon closure rule converges to 1/|det J|, not to 1.

        The final-state functions are complete in Q_f space, so
        sum_v psi_f^v(Q_f) psi_f^v(Q_f') = delta(Q_f - Q_f').  Substituting
        Q_f = J Q_i + K and integrating over Q_i introduces the Jacobian
        1/|det J|, giving sum_v |<v_f|0_i>|^2 = 1/|det J|.  It equals 1 only when
        J is orthogonal, i.e. when both states span the same vibrational
        subspace.

        This matters in practice: two different equilibrium geometries have
        slightly different rotational subspaces, so |det J| deviates from 1 at
        about the 1e-3 level and the raw sum can even exceed 1.  Judging
        completeness against 1 would misreport that as an enumeration error.
        '''
        wi = numpy.array([0.010, 0.018])
        wf = numpy.array([0.013, 0.015])
        th = numpy.deg2rad(31.0)
        rot2 = numpy.array([[math.cos(th), -math.sin(th)],
                            [math.sin(th), math.cos(th)]])

        for scale in (1.0, 0.97, 1.05, 1.20):
            J = rot2.copy()
            J[0] *= scale
            K = numpy.array([0.9, -0.6])
            states, _ = vibronic.enumerate_states(2, 60)
            ov = vibronic.multimode_overlaps(wi, wf, J, K, states)
            total = float((ov ** 2).sum())
            target = 1.0 / abs(numpy.linalg.det(J))
            self.assertAlmostEqual(total / target, 1.0, 12)
            if scale == 1.0:
                self.assertAlmostEqual(total, 1.0, 12)
            else:
                # a genuinely different value, so the test is not vacuous
                self.assertGreater(abs(total - 1.0), 1e-3)

        # and a random non-orthogonal J in 3-D
        rng = numpy.random.RandomState(7)
        J3 = rng.randn(3, 3) * 0.25 + numpy.eye(3)
        wi3 = numpy.array([0.009, 0.013, 0.020])
        wf3 = numpy.array([0.011, 0.012, 0.017])
        states, _ = vibronic.enumerate_states(3, 34)
        ov = vibronic.multimode_overlaps(wi3, wf3, J3, numpy.array([0.5, -0.4, 0.3]),
                                         states)
        self.assertAlmostEqual(float((ov ** 2).sum()) * abs(numpy.linalg.det(J3)),
                               1.0, 10)

    def test_sum_rule_target_reported_through_the_result(self):
        '''The result object must expose the correct target and deficit, so a
        geometric det-J shift is never reported as an enumeration error.'''
        mol = water()
        mf_scf = scf.RHF(mol).run()
        hess = mf_scf.Hessian().kernel()
        mass = mol.atom_mass_list(isotope_avg=True)
        mi = HarmonicModel(mol.atom_charges(), mol.atom_coords(), mass, hess,
                           energy=-75.0, verbose=0)
        c_f = mol.atom_coords().copy()
        c_f[1, 1] *= 1.08
        c_f[2, 1] *= 1.08
        mf2 = HarmonicModel(mol.atom_charges(), c_f, mass, 0.85 * hess,
                            energy=-74.7, verbose=0)
        fc = vibronic.FranckCondon(mi, None, mf2, None)
        res = fc.run(max_quanta=14)

        det = abs(numpy.linalg.det(fc.duschinsky.J))
        self.assertAlmostEqual(res.sum_rule_target, 1.0 / det, 12)
        # non-trivially different from 1
        self.assertGreater(abs(res.sum_rule_target - 1.0), 1e-5)
        # converged against the *target*, not against 1
        self.assertAlmostEqual(res.sum_rule / res.sum_rule_target, 1.0, 6)
        rep = vibronic.analysis.sum_rule_report(res)
        self.assertAlmostEqual(rep['target'], res.sum_rule_target, 14)
        self.assertAlmostEqual(rep['missing'], res.sum_rule_deficit, 14)
        self.assertTrue(rep['adequate'])
        self.assertIn('1/|det J|', res.summary())

    def test_sum_rule_target_defaults_to_one_without_duschinsky(self):
        '''Driven from raw arrays there is no Duschinsky object, so det J is
        unknown and the target must fall back to 1 rather than guessing.'''
        res = vibronic.franck_condon_factors(
            numpy.array([0.010]), numpy.array([0.012]),
            numpy.eye(1), numpy.array([0.7]), max_quanta=30)
        self.assertIsNone(res.duschinsky)
        self.assertEqual(res.sum_rule_target, 1.0)

    def test_max_states_cap_is_recorded(self):
        mol = water()
        mf_scf = scf.RHF(mol).run()
        hess = mf_scf.Hessian().kernel()
        mass = mol.atom_mass_list(isotope_avg=True)
        mi = HarmonicModel(mol.atom_charges(), mol.atom_coords(), mass, hess,
                           energy=-75.0, verbose=0)
        c_f = mol.atom_coords() * 1.03
        mf2 = HarmonicModel(mol.atom_charges(), c_f, mass, 0.9 * hess,
                            energy=-74.8, verbose=0)
        res = vibronic.FranckCondon(mi, None, mf2, None).run(
            max_quanta=8, max_states=25)
        self.assertLessEqual(res.nstate, 25)
        self.assertTrue(res.truncation.get('truncated', False))

    # ------------------------------------------------------------ error paths

    def test_incompatible_states_raise(self):
        h2 = diatomic_model(1.4, 0.35, mass=(1.00794, 1.00794))
        # different element
        d2 = HarmonicModel([1, 3], numpy.array([[0., 0., -0.7], [0., 0., 0.7]]),
                           [1.00794, 6.941],
                           diatomic_model(1.4, 0.35).hessian.reshape(2, 3, 2, 3
                                                                    ).transpose(0, 2, 1, 3),
                           verbose=0)
        self.assertRaises(ValueError, vibronic.FranckCondon(h2, None, d2, None).kernel)

        # different number of atoms
        mol = water()
        mf_scf = scf.RHF(mol).run()
        hess = mf_scf.Hessian().kernel()
        w = HarmonicModel(mol.atom_charges(), mol.atom_coords(),
                          mol.atom_mass_list(isotope_avg=True), hess, verbose=0)
        self.assertRaises(ValueError, vibronic.FranckCondon(h2, None, w, None).kernel)

    def test_hessian_required_with_mole(self):
        mol = water()
        self.assertRaises(ValueError, vibronic.FranckCondon, mol, None, mol, None)

    def test_hessian_rejected_with_prebuilt_model(self):
        m = diatomic_model(2.2, 0.3)
        self.assertRaises(ValueError, vibronic.FranckCondon,
                          m, numpy.zeros((2, 2, 3, 3)), m, None)

    def test_run_rejects_unknown_keyword(self):
        m = diatomic_model(2.2, 0.3)
        fc = vibronic.FranckCondon(m, None, m, None)
        self.assertRaises(AttributeError, fc.run, max_quantaa=3)

    # --------------------------------------------------------------- reporting

    def test_analyze_and_repr_run_clean(self):
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.26, mass=(12.0, 12.0), energy=-19.5)
        fc = vibronic.FranckCondon(mi, None, mf, None, verbose=0)
        fc.run(max_quanta=8)
        self.assertIn('FranckCondon', repr(fc))
        with lib.temporary_env(fc, verbose=0):
            fc.analyze()
        ana = vibronic.analysis.huang_rhys_analysis(fc.duschinsky)
        self.assertAlmostEqual(
            ana['total_reorganization_energy'],
            float(numpy.sum(fc.duschinsky.huang_rhys * fc.duschinsky.freq_f)), 14)
        contrib = vibronic.analysis.mode_contributions(fc.result)
        self.assertEqual(len(contrib['mode']), 1)

    def test_determinism(self):
        mi = diatomic_model(2.2, 0.30, mass=(12.0, 12.0), energy=-20.0)
        mf = diatomic_model(2.5, 0.26, mass=(12.0, 12.0), energy=-19.5)
        a = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=9)
        b = vibronic.FranckCondon(mi, None, mf, None).run(max_quanta=9)
        self.assertTrue(numpy.array_equal(a.states, b.states))
        self.assertAlmostEqual(abs(a.fcf - b.fcf).max(), 0.0, 15)
        self.assertAlmostEqual(abs(a.energies - b.energies).max(), 0.0, 15)


if __name__ == '__main__':
    print('Full Tests for pyscf.vibronic workflow')
    unittest.main()
