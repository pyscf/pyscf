#!/usr/bin/env python

'''
Harmonic Franck-Condon absorption spectrum.

The high-level driver takes two states -- each at its own optimised geometry,
with a Hessian computed there -- and produces vibronic stick and broadened
spectra.  For ABSORPTION the state the molecule starts in (the lower state) is
passed as `initial`.

The two states here are two SCF states of water at different geometries, used as
a cheap stand-in.  In production the second state would come from an
excited-state geometry optimisation and Hessian; pyscf.vibronic only ever sees
geometries, Hessians and energies, so it is independent of the method that
produced them (TDDFT, CASSCF, EOM-CC, or an external program).

Everything reported is harmonic and within the Condon approximation.  No
anharmonic or Herzberg-Teller correction is applied, so these numbers are not
converged spectroscopic predictions.
'''

import numpy
from pyscf import gto, scf, vibronic
from pyscf.vibronic import units

basis = '631g'

# ---- ground state ---------------------------------------------------------
mol_g = gto.M(atom = '''O   0.   0.       0.1173
                        H   0.   0.7572  -0.4692
                        H   0.  -0.7572  -0.4692''', basis = basis)
mf_g = mol_g.RHF().run()
hess_g = mf_g.Hessian().kernel()

# ---- "excited" state: longer bonds, wider angle, higher energy ------------
mol_e = gto.M(atom = '''O   0.   0.       0.1500
                        H   0.   0.8400  -0.6000
                        H   0.  -0.8400  -0.6000''', basis = basis)
mf_e = mol_e.RHF().run()
hess_e = mf_e.Hessian().kernel()

# A plausible adiabatic excitation energy, in Hartree.  Only the difference
# final_energy - initial_energy matters.
e_exc = 7.5 / units.HARTREE2EV

fc = vibronic.FranckCondon(mol_g, hess_g, mol_e, hess_e,
                           initial_energy=0.0, final_energy=e_exc)

# Enumeration controls.  max_quanta is the total number of quanta allowed in the
# final state; max_modes_excited caps how many modes may be excited at once.
fc.max_quanta = 8
fc.max_modes_excited = 3
fc.allow_imaginary = True          # the model geometries are not stationary
result = fc.run()

print('\nadiabatic energy   %8.4f eV' % units.au2ev(result.e_adiabatic))
print('0-0 energy         %8.4f eV = %8.1f cm^-1 = %6.1f nm'
      % (units.au2ev(result.e_00), units.au2wavenumber(result.e_00),
         units.au2nm(result.e_00)))
print('ZPE initial/final  %8.5f / %8.5f Eh' % (result.zpe_i, result.zpe_f))
print('states enumerated  %d' % result.nstate)
print('sum rule           %.8f   (1 - sum = %.2e)'
      % (result.sum_rule, 1 - result.sum_rule))

# The sum rule is the honest measure of whether the enumeration is complete: a
# finite max_quanta always leaves some intensity out.  Check convergence.
print('\nsum-rule convergence:')
for mq in (2, 4, 6, 8, 10):
    r = vibronic.FranckCondon(mol_g, hess_g, mol_e, hess_e,
                              initial_energy=0.0, final_energy=e_exc)
    r.allow_imaginary = True
    r.verbose = 0
    print('  max_quanta = %2d  ->  sum rule = %.8f' % (mq, r.run(max_quanta=mq).sum_rule))

# ---- stick spectrum -------------------------------------------------------
# The default intensity carries the E^1 factor appropriate to an absorption
# cross-section.  Use line_strength=True for the bare Franck-Condon factors.
sticks = result.stick_spectrum(kind='absorption')
print('\n%d lines, strongest first:' % len(sticks.energies))
cm = sticks.to_unit('cm-1')
order = numpy.argsort(-sticks.intensities)[:12]
print('  %12s %10s %12s   %s' % ('E/cm^-1', 'E/eV', 'rel. int.', 'assignment'))
for i in order:
    occ = result.states[i]
    label = ' '.join('%d^%d' % (k, occ[k]) for k in numpy.nonzero(occ)[0]) or '0-0'
    print('  %12.1f %10.4f %12.4e   %s'
          % (cm.energies[i], units.au2ev(sticks.energies[i]),
             sticks.intensities[i] / sticks.intensities.max(), label))

# ---- broadened spectrum ---------------------------------------------------
# `width` is the FWHM, expressed in `unit`.  StickSpectrum.broaden() returns a
# BroadenedSpectrum object; the lower-level vibronic.broaden() function returns
# a plain (grid, signal) tuple.
spec = cm.broaden(profile='gaussian', width=400.0, unit='cm-1', npoints=4000)
print('\nbroadened: %d points over [%.0f, %.0f] cm^-1' % (len(spec.x), spec.x[0], spec.x[-1]))
print('area / total stick intensity = %.6f'
      % (vibronic.trapezoid(spec.y, spec.x) / sticks.intensities.sum()))
print('band maximum at %.1f cm^-1 = %.4f eV'
      % (spec.x[spec.y.argmax()], units.au2ev(units.wavenumber2au(spec.x[spec.y.argmax()]))))

# A Lorentzian profile is also available.  Note its 1/x^2 tails: any modest
# truncation loses a noticeable fraction of the area, so no cutoff is applied
# by default.
lor = cm.broaden(profile='lorentzian', width=400.0, unit='cm-1', npoints=4000)
print('lorentzian band maximum at %.1f cm^-1' % lor.x[lor.y.argmax()])

# ---- diagnostics ----------------------------------------------------------
fc.analyze(nline=10)

# Optional plotting; matplotlib is NOT a dependency of pyscf.vibronic.
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('\n(matplotlib not available; skipping the plot)')
else:
    plt.vlines(cm.energies, 0, sticks.intensities / sticks.intensities.max(),
               color='0.7', lw=1)
    plt.plot(spec.x, spec.y / spec.y.max(), 'k-')
    plt.xlabel(r'photon energy / cm$^{-1}$')
    plt.ylabel('relative intensity')
    plt.title('Harmonic Franck-Condon absorption')
    plt.tight_layout()
    plt.savefig('fc_absorption.png', dpi=120)
    print('\nwrote fc_absorption.png')
