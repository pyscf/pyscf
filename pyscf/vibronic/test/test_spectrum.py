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

'''Stick spectra, lineshape profiles and broadening.'''

import math
import unittest

import numpy
import scipy.integrate

from pyscf.vibronic import franck_condon as fc
from pyscf.vibronic import spectrum as sp
from pyscf.vibronic import units


def rot2(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return numpy.array([[c, -s], [s, c]])


FREQ_I = numpy.array([0.0080, 0.0150])
FREQ_F = numpy.array([0.0110, 0.0190])
K = numpy.array([0.55, -0.35])
J = rot2(25.0)


def absorption_result(max_quanta=3, e_adiabatic=0.20):
    '''An FC result with Delta E > 0 throughout (initial = lower state).'''
    return fc.franck_condon_factors(FREQ_I, FREQ_F, J, K, e_adiabatic=e_adiabatic,
                                    max_quanta=max_quanta, verbose=0)


def emission_result(max_quanta=3, e_adiabatic=-0.20):
    '''An FC result with Delta E < 0 throughout (initial = the EXCITED state).'''
    return fc.franck_condon_factors(FREQ_I, FREQ_F, J, K, e_adiabatic=e_adiabatic,
                                    max_quanta=max_quanta, verbose=0)


def _measured_fwhm(func, x0, fwhm, unit_span=6.0, npoints=4000001):
    '''Numerically locate the half-maximum crossings of a profile.'''
    x = numpy.linspace(x0 - unit_span * fwhm, x0 + unit_span * fwhm, npoints)
    y = func(x, x0, fwhm)
    half = 0.5 * y.max()
    above = numpy.where(y >= half)[0]
    lo, hi = above[0], above[-1]

    def interp(i, j):
        # linear interpolation of the half-max crossing between grid points i, j
        return x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i])
    left = interp(lo, lo - 1)
    right = interp(hi, hi + 1)
    return right - left


class ProfileTests(unittest.TestCase):

    def test_gaussian_area_normalised(self):
        for fwhm in (0.3, 1.0, 7.5, 250.0):
            area, err = scipy.integrate.quad(
                lambda t: sp.gaussian_profile(t, 2.0, fwhm), -numpy.inf, numpy.inf)
            # measured |area - 1| <= 3e-13
            self.assertAlmostEqual(area, 1.0, 10)

    def test_lorentzian_area_normalised(self):
        for fwhm in (0.3, 1.0, 7.5, 250.0):
            area, err = scipy.integrate.quad(
                lambda t: sp.lorentzian_profile(t, -1.5, fwhm), -numpy.inf, numpy.inf)
            # measured |area - 1| <= 2e-11
            self.assertAlmostEqual(area, 1.0, 10)

    def test_fwhm_is_exactly_the_fwhm_analytic(self):
        '''The half maximum must sit exactly at x0 +- fwhm/2.'''
        for func in (sp.gaussian_profile, sp.lorentzian_profile):
            for fwhm in (0.5, 3.0, 120.0):
                peak = func(1.25, 1.25, fwhm)
                for sign in (-1.0, 1.0):
                    half = func(1.25 + sign * 0.5 * fwhm, 1.25, fwhm)
                    self.assertAlmostEqual(half / peak, 0.5, 13)

    def test_fwhm_measured_numerically(self):
        '''Independently *measure* the width from the sampled profile.'''
        for func in (sp.gaussian_profile, sp.lorentzian_profile):
            for fwhm in (1.0, 40.0):
                got = _measured_fwhm(func, 3.0, fwhm)
                # measured relative error <= 2e-9 on a 4e6-point grid
                self.assertAlmostEqual(got / fwhm, 1.0, 7)

    def test_gaussian_matches_references_explicit_form(self):
        '''REFERENCES.md Eq. (8.5): g = (2/G) sqrt(ln2/pi) exp(-4 ln2 (x-x0)^2/G^2).'''
        x = numpy.linspace(-4.0, 6.0, 501)
        for fwhm in (0.7, 3.0, 55.0):
            ref = (2.0 / fwhm) * math.sqrt(math.log(2.0) / math.pi) * numpy.exp(
                -4.0 * math.log(2.0) * (x - 1.1) ** 2 / fwhm ** 2)
            got = sp.gaussian_profile(x, 1.1, fwhm)
            self.assertAlmostEqual(float(numpy.abs(got - ref).max()), 0.0, 14)
        # and the sigma relation quoted there
        self.assertAlmostEqual(sp.GAUSS_FWHM_TO_SIGMA, 2.354820045030949, 12)

    def test_lorentzian_matches_references_explicit_form(self):
        '''REFERENCES.md Eq. (8.6): l = (1/pi) (G/2)/((x-x0)^2 + (G/2)^2).'''
        x = numpy.linspace(-10.0, 10.0, 401)
        for fwhm in (0.9, 12.0):
            g2 = 0.5 * fwhm
            ref = (1.0 / math.pi) * g2 / ((x - 0.4) ** 2 + g2 ** 2)
            got = sp.lorentzian_profile(x, 0.4, fwhm)
            self.assertAlmostEqual(float(numpy.abs(got - ref).max()), 0.0, 15)

    def test_lorentzian_tails_are_heavy(self):
        '''Documents why the Lorentzian cutoff must be wide (or disabled).

        Truncating at +-n*FWHM keeps (2/pi) arctan(2n) of the area, so the loss
        is 1 - (2/pi) arctan(2n) ~ 1/(pi n).  Measured: n = 5 loses 0.063451
        (1/(pi n) = 0.063662); n = 50 loses 0.00636599 (1/(pi n) = 0.00636620).
        '''
        fwhm = 1.0
        for n in (5.0, 50.0):
            area, _ = scipy.integrate.quad(
                lambda t: sp.lorentzian_profile(t, 0.0, fwhm), -n * fwhm, n * fwhm)
            lost = 1.0 - area
            exact = 1.0 - (2.0 / math.pi) * math.atan(2.0 * n)
            self.assertAlmostEqual(lost, exact, 12)
            self.assertAlmostEqual(lost, 1.0 / (math.pi * n), 3)
        # the Gaussian at 5 FWHM has lost nothing measurable
        area, _ = scipy.integrate.quad(
            lambda t: sp.gaussian_profile(t, 0.0, 1.0), -5.0, 5.0)
        self.assertAlmostEqual(area, 1.0, 12)

    def test_bad_width_raises(self):
        self.assertRaises(ValueError, sp.gaussian_profile, 0.0, 0.0, 0.0)
        self.assertRaises(ValueError, sp.gaussian_profile, 0.0, 0.0, -1.0)
        self.assertRaises(ValueError, sp.lorentzian_profile, 0.0, 0.0, -3.0)


class StickTests(unittest.TestCase):

    def test_absorption_energies_positive_and_signed_convention(self):
        res = absorption_result()
        st = sp.stick_spectrum(res, kind='absorption')
        self.assertEqual(st.kind, 'absorption')
        self.assertTrue(numpy.all(st.energies > 0.0))
        # photon energy = +Delta E
        self.assertAlmostEqual(float(st.energies.min()), float(res.energies.min()), 14)
        self.assertAlmostEqual(float(st.energies.max()), float(res.energies.max()), 14)
        # sorted ascending in photon energy
        self.assertTrue(numpy.all(numpy.diff(st.energies) >= -1e-18))
        self.assertEqual(st.assignments.shape, (st.nline, 2))

    def test_emission_energies_positive_from_negative_delta_e(self):
        res = emission_result()
        self.assertTrue(numpy.all(res.energies < 0.0))
        st = sp.stick_spectrum(res, kind='emission')
        self.assertTrue(numpy.all(st.energies > 0.0))
        # photon energy = -Delta E
        self.assertAlmostEqual(float(st.energies.max()), -float(res.energies.min()), 14)

    def test_wrongly_ordered_states_raise(self):
        '''Requesting the wrong kind for the sign of Delta E must raise.'''
        absorp = absorption_result()
        with self.assertRaises(ValueError) as ctx:
            sp.stick_spectrum(absorp, kind='emission')
        msg = str(ctx.exception)
        self.assertIn('EXCITED', msg)
        self.assertIn('Delta E', msg)

        emis = emission_result()
        with self.assertRaises(ValueError) as ctx:
            sp.stick_spectrum(emis, kind='absorption')
        self.assertIn('absorption', str(ctx.exception))

        self.assertRaises(ValueError, sp.stick_spectrum, absorp, 'fluorescence')

    def test_e1_prefactor_for_absorption(self):
        res = absorption_result()
        bare = sp.stick_spectrum(res, kind='absorption', line_strength=True)
        weighted = sp.stick_spectrum(res, kind='absorption')
        self.assertTrue(bare.line_strength)
        self.assertFalse(weighted.line_strength)
        self.assertAlmostEqual(float(numpy.abs(bare.energies - weighted.energies).max()),
                               0.0, 15)
        ratio = weighted.intensities / bare.intensities
        self.assertAlmostEqual(float(numpy.abs(ratio - weighted.energies).max()), 0.0, 14)

    def test_e3_prefactor_for_emission(self):
        res = emission_result()
        bare = sp.stick_spectrum(res, kind='emission', line_strength=True)
        weighted = sp.stick_spectrum(res, kind='emission')
        ratio = weighted.intensities / bare.intensities
        self.assertAlmostEqual(float(numpy.abs(ratio - weighted.energies ** 3).max()),
                               0.0, 14)

    def test_line_strength_is_bare_fcf(self):
        '''line_strength=True is the p=0 Franck-Condon profile (REFERENCES 8.2).'''
        res = absorption_result()
        st = sp.stick_spectrum(res, kind='absorption', line_strength=True)
        # no dipole, no temperature: intensity == FCF exactly
        order = numpy.argsort(res.energies, kind='stable')
        self.assertAlmostEqual(float(numpy.abs(st.intensities - res.fcf[order]).max()),
                               0.0, 15)
        self.assertAlmostEqual(st.total_intensity, res.sum_rule, 12)

    def test_transition_dipole_scaling(self):
        res = absorption_result()
        base = sp.stick_spectrum(res, kind='absorption', line_strength=True)
        scalar = sp.stick_spectrum(res, kind='absorption', line_strength=True,
                                   transition_dipole=0.4)
        vector = sp.stick_spectrum(res, kind='absorption', line_strength=True,
                                   transition_dipole=[0.3, -0.1, 0.2])
        self.assertAlmostEqual(scalar.total_intensity / base.total_intensity, 0.16, 12)
        mu2 = 0.3 ** 2 + 0.1 ** 2 + 0.2 ** 2
        self.assertAlmostEqual(vector.total_intensity / base.total_intensity, mu2, 12)

    def test_intensity_threshold(self):
        res = absorption_result(max_quanta=6)
        full = sp.stick_spectrum(res, kind='absorption', line_strength=True)
        cut = sp.stick_spectrum(res, kind='absorption', line_strength=True,
                                intensity_threshold=1e-5)
        self.assertLess(cut.nline, full.nline)
        self.assertTrue(numpy.all(cut.intensities >= 1e-5))

    def test_unit_round_trip(self):
        res = absorption_result()
        st = sp.stick_spectrum(res, kind='absorption')
        cm = st.to_unit('cm-1')
        ev = cm.to_unit('ev')
        back = ev.to_unit('au')
        self.assertEqual(cm.unit, 'cm-1')
        self.assertAlmostEqual(float(numpy.abs(back.energies - st.energies).max()), 0.0, 15)
        # spot-check the actual conversion factors
        self.assertAlmostEqual(float(cm.energies[0]),
                               float(units.au2wavenumber(st.energies[0])), 8)
        self.assertAlmostEqual(float(ev.energies[0]),
                               float(units.au2ev(st.energies[0])), 12)
        # intensities are integrated quantities: unchanged by a unit change
        self.assertAlmostEqual(cm.total_intensity, st.total_intensity, 15)
        self.assertRaises(ValueError, st.to_unit, 'furlongs')
        # 'nm' is deliberately rejected for linear conversion
        self.assertRaises(ValueError, st.to_unit, 'nm')

    def test_merge_preserves_total_intensity(self):
        res = absorption_result(max_quanta=5)
        st = sp.stick_spectrum(res, kind='absorption')
        # measured: 21 distinct lines, minimum spacing 2.000e-03 Eh; merge_tol
        # = 4e-3 Eh collapses them to 10 lines
        merged = sp.stick_spectrum(res, kind='absorption', merge_tol=4e-3)
        self.assertEqual(st.nline, 21)
        self.assertEqual(merged.nline, 10)
        self.assertLess(merged.nline, st.nline)
        self.assertAlmostEqual(merged.total_intensity, st.total_intensity, 14)
        # merging is idempotent-ish: a zero/None tolerance changes nothing
        same = sp.stick_spectrum(res, kind='absorption', merge_tol=0.0)
        self.assertEqual(same.nline, st.nline)

    def test_merge_exactly_degenerate_lines(self):
        '''Two exactly coincident sticks must merge into one, summing intensity.'''
        freq_i = numpy.array([0.010, 0.010])
        freq_f = numpy.array([0.012, 0.012])       # exact degeneracy
        res = fc.franck_condon_factors(freq_i, freq_f, rot2(20.0),
                                       numpy.array([0.4, -0.3]),
                                       e_adiabatic=0.2, max_quanta=3, verbose=0)
        st = sp.stick_spectrum(res, kind='absorption')
        merged = sp.stick_spectrum(res, kind='absorption', merge_tol=1e-12)
        # total quanta 0,1,2,3 -> 4 distinct energies
        self.assertEqual(merged.nline, 4)
        self.assertLess(merged.nline, st.nline)
        self.assertAlmostEqual(merged.total_intensity, st.total_intensity, 15)

    def test_determinism(self):
        res = absorption_result(max_quanta=4)
        a = sp.stick_spectrum(res, kind='absorption', merge_tol=4e-3)
        b = sp.stick_spectrum(res, kind='absorption', merge_tol=4e-3)
        self.assertEqual(a.energies.tobytes(), b.energies.tobytes())
        self.assertEqual(a.intensities.tobytes(), b.intensities.tobytes())
        self.assertEqual(a.assignments.tobytes(), b.assignments.tobytes())

    def test_result_method_delegates(self):
        res = absorption_result()
        st = res.stick_spectrum(kind='absorption')
        self.assertIsInstance(st, sp.StickSpectrum)
        self.assertEqual(st.nline, res.nstate)
        self.assertIn('StickSpectrum', repr(st))


class BroadeningTests(unittest.TestCase):

    def test_gaussian_conserves_area(self):
        res = absorption_result(max_quanta=4)
        st = sp.stick_spectrum(res, kind='absorption')
        grid, signal = sp.broaden(st.energies, st.intensities, profile='gaussian',
                                  width=300.0, unit='cm-1', npoints=200001,
                                  padding=8000.0)
        area = float(numpy.trapz(signal, grid))
        # measured relative error 2e-15
        self.assertAlmostEqual(area / st.total_intensity, 1.0, 8)

    def test_gaussian_cutoff_loses_nothing(self):
        res = absorption_result(max_quanta=3)
        st = sp.stick_spectrum(res, kind='absorption')
        kw = dict(profile='gaussian', width=200.0, unit='cm-1', npoints=120001,
                  padding=8000.0)
        g1, s1 = sp.broaden(st.energies, st.intensities, cutoff=None, **kw)
        g2, s2 = sp.broaden(st.energies, st.intensities, cutoff=5.0, **kw)
        self.assertTrue(numpy.array_equal(g1, g2))
        # measured max |difference| / max signal = 4e-32
        self.assertLess(float(numpy.abs(s1 - s2).max()) / float(s1.max()), 1e-20)

    def test_lorentzian_area_needs_a_wide_grid(self):
        '''The Lorentzian only conserves area on a very wide grid (heavy tails).'''
        st = sp.StickSpectrum([0.2], [1.0], kind='absorption')
        for padding, places in ((1e5, 2), (1e7, 4)):
            grid, signal = sp.broaden(st.energies, st.intensities,
                                      profile='lorentzian', width=300.0, unit='cm-1',
                                      npoints=400001, padding=padding)
            area = float(numpy.trapz(signal, grid))
            self.assertAlmostEqual(area, 1.0, places)
        # by contrast the Gaussian is converged with modest padding
        grid, signal = sp.broaden(st.energies, st.intensities, profile='gaussian',
                                  width=300.0, unit='cm-1', npoints=200001,
                                  padding=5000.0)
        self.assertAlmostEqual(float(numpy.trapz(signal, grid)), 1.0, 10)

    def test_single_stick_recovers_the_profile(self):
        '''One unit stick broadens to exactly the normalised profile.'''
        e0 = 0.25
        for profile, func in (('gaussian', sp.gaussian_profile),
                              ('lorentzian', sp.lorentzian_profile)):
            grid, signal = sp.broaden([e0], [1.0], profile=profile, width=400.0,
                                      unit='cm-1', npoints=20001, padding=6000.0,
                                      cutoff=None)
            ref = func(grid, float(units.au2wavenumber(e0)), 400.0)
            self.assertAlmostEqual(float(numpy.abs(signal - ref).max()), 0.0, 14)

    def test_explicit_grid(self):
        st = sp.StickSpectrum([0.2, 0.21], [1.0, 2.0], kind='absorption')
        g = numpy.linspace(40000.0, 50000.0, 5001)
        grid, signal = sp.broaden(st.energies, st.intensities, grid=g, width=200.0,
                                  unit='cm-1')
        self.assertTrue(numpy.array_equal(grid, g))
        self.assertEqual(signal.shape, g.shape)
        self.assertRaises(ValueError, sp.broaden, st.energies, st.intensities,
                          grid=[3.0, 1.0, 2.0], width=100.0, unit='cm-1')

    def test_broaden_method_returns_container(self):
        res = absorption_result(max_quanta=4)
        st = sp.stick_spectrum(res, kind='absorption')
        bs = st.broaden(profile='gaussian', width=250.0, unit='cm-1', npoints=100001,
                        padding=8000.0)
        self.assertIsInstance(bs, sp.BroadenedSpectrum)
        self.assertEqual(bs.unit, 'cm-1')
        self.assertEqual(bs.kind, 'absorption')
        self.assertAlmostEqual(bs.area / st.total_intensity, 1.0, 8)
        self.assertIn('BroadenedSpectrum', repr(bs))
        # broadening from a stick spectrum stored in another unit is identical
        bs2 = st.to_unit('ev').broaden(profile='gaussian', width=250.0, unit='cm-1',
                                      npoints=100001, padding=8000.0)
        self.assertAlmostEqual(float(numpy.abs(bs2.y - bs.y).max()) / float(bs.y.max()),
                               0.0, 10)

    def test_broadened_unit_conversion_preserves_area(self):
        res = absorption_result(max_quanta=3)
        st = sp.stick_spectrum(res, kind='absorption')
        bs = st.broaden(profile='gaussian', width=300.0, unit='cm-1', npoints=100001,
                        padding=8000.0)
        ev = bs.to_unit('ev')
        self.assertEqual(ev.unit, 'ev')
        self.assertAlmostEqual(ev.area / bs.area, 1.0, 12)
        back = ev.to_unit('cm-1')
        self.assertAlmostEqual(float(numpy.abs(back.x - bs.x).max()), 0.0, 8)

    def test_broaden_determinism(self):
        res = absorption_result(max_quanta=4)
        st = sp.stick_spectrum(res, kind='absorption')
        kw = dict(profile='gaussian', width=300.0, unit='cm-1', npoints=20001,
                  padding=6000.0)
        g1, s1 = sp.broaden(st.energies, st.intensities, **kw)
        g2, s2 = sp.broaden(st.energies, st.intensities, **kw)
        self.assertEqual(g1.tobytes(), g2.tobytes())
        self.assertEqual(s1.tobytes(), s2.tobytes())

    def test_broaden_input_validation(self):
        self.assertRaises(ValueError, sp.broaden, [0.2], [1.0], profile='voigt',
                          width=100.0)
        self.assertRaises(ValueError, sp.broaden, [0.2], [1.0], width=0.0)
        self.assertRaises(ValueError, sp.broaden, [0.2], [1.0, 2.0], width=100.0)
        self.assertRaises(ValueError, sp.broaden, [0.2], [1.0], width=100.0,
                          unit='parsecs')
        self.assertRaises(ValueError, sp.broaden, [0.2], [1.0], width=100.0, npoints=1)

    def test_empty_spectrum(self):
        grid, signal = sp.broaden([], [], width=100.0, unit='cm-1', npoints=101)
        self.assertEqual(grid.shape, (101,))
        self.assertTrue(numpy.all(signal == 0.0))

    def test_no_matplotlib_import(self):
        '''The numerical modules must never pull in a plotting library.'''
        import sys
        self.assertNotIn('matplotlib', sys.modules)
        with open(sp.__file__) as fh:
            src = fh.read()
        self.assertNotIn('matplotlib', src)
        self.assertNotIn('pyplot', src)


class TemperatureTests(unittest.TestCase):

    def test_populations_enter_the_intensities(self):
        res = fc.franck_condon_factors(FREQ_I, FREQ_F, J, K, e_adiabatic=0.20,
                                       max_quanta=3, temperature=3000.0,
                                       max_quanta_init=2, verbose=0)
        st = sp.stick_spectrum(res, kind='absorption', line_strength=True)
        order = numpy.argsort(res.energies, kind='stable')
        ref = (res.populations * res.fcf)[order]
        self.assertAlmostEqual(float(numpy.abs(st.intensities - ref).max()), 0.0, 15)
        self.assertEqual(st.temperature, 3000.0)
        # hot bands must appear to the red of the origin
        self.assertLess(float(st.energies.min()), res.e_00)
        # and the initial-state assignments are carried through
        self.assertGreater(int((st.init_assignments.sum(axis=1) > 0).sum()), 0)


if __name__ == '__main__':
    print('Full Tests for pyscf.vibronic.spectrum')
    unittest.main()
