#!/usr/bin/env python

'''
The low-level API: Franck-Condon factors from raw arrays.

pyscf.vibronic is method-independent.  The overlap kernels need only

    freq_i (n_i,)   initial-state angular frequencies, Eh (hbar = 1)
    freq_f (n_f,)   final-state angular frequencies, Eh
    J      (n_f,n_i) Duschinsky matrix, rows = FINAL modes, cols = INITIAL modes
    K      (n_f,)   displacement, bohr sqrt(m_e)

so a model Hamiltonian, data from another program, or a fitted spectroscopic
model can be used directly, with no Mole and no Hessian anywhere in sight.

This example
  1. checks the analytic one-dimensional limits,
  2. builds a two-mode model with genuine Duschinsky mixing,
  3. shows what the closure rule really converges to,
  4. uses the enumeration and broadening functions directly.
'''

import math
import numpy
from pyscf import vibronic
from pyscf.vibronic import units

# =====================================================================
# 1. One dimension: the displaced oscillator is exactly Poissonian
# =====================================================================
# For equal frequencies and a displacement d, |<n|0>|^2 = e^-S S^n / n! with
# the Huang-Rhys factor S = omega * d^2 / 2.
omega = 0.009
d = 4.0
S = vibronic.huang_rhys(omega, d)
print('1-D displaced oscillator, S = %.6f' % S)
print('  %3s %20s %20s %10s' % ('n', '|<n|0>|^2', 'e^-S S^n/n!', 'diff'))
worst = 0.0
for n in range(6):
    fcf = vibronic.overlap_1d(n, 0, omega, omega, d) ** 2
    exact = math.exp(-S) * S ** n / math.factorial(n)
    worst = max(worst, abs(fcf - exact))
    print('  %3d %20.15f %20.15f %10.1e' % (n, fcf, exact, abs(fcf - exact)))
print('  max deviation over n = 0..20: %.2e'
      % max(abs(vibronic.overlap_1d(n, 0, omega, omega, d) ** 2
                - math.exp(-S) * S ** n / math.factorial(n)) for n in range(21)))

# A pure frequency change with no displacement: only even n survive, by parity.
wa, wb = 0.008, 0.017
print('\n1-D frequency change, no displacement (%.3f -> %.3f Eh)' % (wa, wb))
print('  <0|0>            = %.15f' % vibronic.overlap_1d(0, 0, wa, wb, 0.0))
print('  closed form      = %.15f' % math.sqrt(2 * math.sqrt(wa * wb) / (wa + wb)))
print('  max odd |<n|0>|  = %.2e'
      % max(abs(vibronic.overlap_1d(n, 0, wa, wb, 0.0)) for n in (1, 3, 5, 7)))

# =====================================================================
# 2. Two modes with genuine Duschinsky mixing
# =====================================================================
theta = numpy.deg2rad(35.0)
J = numpy.array([[math.cos(theta), -math.sin(theta)],
                 [math.sin(theta),  math.cos(theta)]])
freq_i = numpy.array([0.0100, 0.0180])      # Eh
freq_f = numpy.array([0.0130, 0.0150])
K = numpy.array([2.5, -1.5])                # bohr sqrt(m_e)

print('\n2-mode model: J = rotation(35 deg), |det J| = %.12f'
      % abs(numpy.linalg.det(J)))
print('  freq_i [cm^-1] %s' % numpy.round(units.au2wavenumber(freq_i), 1))
print('  freq_f [cm^-1] %s' % numpy.round(units.au2wavenumber(freq_f), 1))
print('  Huang-Rhys     %s' % numpy.round(vibronic.huang_rhys(freq_f, K), 4))

# The adiabatic energy is the only place an electronic energy enters.
result = vibronic.franck_condon_factors(freq_i, freq_f, J, K,
                                        e_adiabatic=0.15, max_quanta=12)
print('\n%s' % result.summary())

print('\nstrongest lines:')
weight = result.fcf
for idx in numpy.argsort(-weight)[:8]:
    occ = result.states[idx]
    label = ' '.join('%d^%d' % (k, occ[k]) for k in numpy.nonzero(occ)[0]) or '0-0'
    print('  %-12s  E = %9.1f cm^-1   FCF = %.6e'
          % (label, units.au2wavenumber(result.energies[idx]), result.fcf[idx]))

# Switching the rotation off (J = I) changes the whole line pattern, not just a
# scale factor.  For these modest Huang-Rhys factors the 0-0 line moves by only
# a couple of percent, but individual weak lines change by much more -- which is
# why a product of independent one-dimensional overlaps is not a substitute for
# the general treatment.
no_mix = vibronic.franck_condon_factors(freq_i, freq_f, numpy.eye(2), K,
                                        e_adiabatic=0.15, max_quanta=12)
key = {tuple(s): f for s, f in zip(no_mix.states, no_mix.fcf)}
ratios = numpy.array([result.fcf[i] / key[tuple(result.states[i])]
                      for i in range(result.nstate)
                      if key.get(tuple(result.states[i]), 0.0) > 1e-14])
print('\nsame model with J = I (mixing switched off):')
print('  0-0 FCF with mixing    %.6f' % result.fcf[result.states.sum(axis=1).argmin()])
print('  0-0 FCF without mixing %.6f' % no_mix.fcf[no_mix.states.sum(axis=1).argmin()])
print('  per-line FCF ratio (mixed / unmixed): min %.3f, max %.3f over %d lines'
      % (ratios.min(), ratios.max(), len(ratios)))

# =====================================================================
# 3. What the closure rule actually converges to
# =====================================================================
# sum_v |<v_f|0_i>|^2 = 1/|det J|, which is 1 only for orthogonal J.
print('\nclosure rule vs max_quanta (orthogonal J, so the target is exactly 1):')
for mq in (2, 4, 8, 16, 30):
    states, info = vibronic.enumerate_states(2, mq)
    ov = vibronic.multimode_overlaps(freq_i, freq_f, J, K, states)
    print('  max_quanta = %2d  %6d states  sum = %.12f  deficit = %.2e'
          % (mq, len(states), (ov ** 2).sum(), 1.0 - (ov ** 2).sum()))

# Now deliberately break the orthogonality of J.
J_skew = J.copy()
J_skew[0] *= 1.20
states, _ = vibronic.enumerate_states(2, 40)
ov = vibronic.multimode_overlaps(freq_i, freq_f, J_skew, K, states)
print('\nnon-orthogonal J (|det J| = %.6f):' % abs(numpy.linalg.det(J_skew)))
print('  sum = %.12f   1/|det J| = %.12f'
      % ((ov ** 2).sum(), 1.0 / abs(numpy.linalg.det(J_skew))))
print('  -> comparing this sum against 1 would be a mistake')

# =====================================================================
# 4. Enumeration and broadening used directly
# =====================================================================
states, info = vibronic.enumerate_states(4, max_quanta=6, max_modes_excited=2,
                                         max_states=40)
print('\nenumerate_states(4 modes, max_quanta=6, max_modes_excited=2, max_states=40)')
print('  returned %d states; bookkeeping: %s' % (len(states), info))
print('  deterministic ordering, first five rows:\n%s' % states[:5])

sticks = result.stick_spectrum(kind='absorption')
grid, signal = vibronic.broaden(sticks.energies, sticks.intensities,
                                profile='gaussian', width=250.0, unit='cm-1',
                                npoints=4000)
print('\nbroaden(): %d points, area/total = %.8f'
      % (len(grid), vibronic.trapezoid(signal, grid) / sticks.intensities.sum()))

# The profiles themselves are area-normalised and parameterised by FWHM.
x = numpy.linspace(-5000, 5000, 200001)
for name, fn in (('gaussian', vibronic.gaussian_profile),
                 ('lorentzian', vibronic.lorentzian_profile)):
    y = fn(x, 0.0, 500.0)
    half = y.max() / 2
    above = x[y >= half]
    print('  %-11s area = %.10f   measured FWHM = %.2f (requested 500)'
          % (name, vibronic.trapezoid(y, x), above[-1] - above[0]))
