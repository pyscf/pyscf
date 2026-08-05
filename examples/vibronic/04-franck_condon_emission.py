#!/usr/bin/env python

'''
Harmonic Franck-Condon emission (fluorescence) spectrum, and the Stokes shift.

The sign convention is the important part.  pyscf.vibronic always stores the
signed transition energy

    Delta E = E_final(v_f) - E_initial(v_i),

so which spectrum you get is decided by which state you pass as `initial`:

  * absorption -- `initial` is the LOWER state, Delta E > 0, photon = +Delta E,
    intensity carries an E^1 factor (a cross-section);
  * emission   -- `initial` is the UPPER (excited) state, Delta E < 0,
    photon = -Delta E, intensity carries an E^3 factor (a rate).

Requesting a `kind` that contradicts the sign raises ValueError rather than
silently producing negative photon energies.

Both spectra share the same 0-0 energy (up to that sign), and the vibrational
progressions run in opposite directions from it: absorption builds on the
FINAL-state frequencies, emission on the (now final) ground-state frequencies.
That asymmetry, plus the reorganisation energy, is the origin of the Stokes
shift.
'''

import numpy
from pyscf import gto, scf, vibronic
from pyscf.vibronic import units

basis = '631g'

mol_g = gto.M(atom = '''O   0.   0.       0.1173
                        H   0.   0.7572  -0.4692
                        H   0.  -0.7572  -0.4692''', basis = basis)
mf_g = mol_g.RHF().run()
hess_g = mf_g.Hessian().kernel()

mol_e = gto.M(atom = '''O   0.   0.       0.1500
                        H   0.   0.8400  -0.6000
                        H   0.  -0.8400  -0.6000''', basis = basis)
mf_e = mol_e.RHF().run()
hess_e = mf_e.Hessian().kernel()

e_exc = 7.5 / units.HARTREE2EV

# ---- absorption: ground state is `initial` --------------------------------
absorp = vibronic.FranckCondon(mol_g, hess_g, mol_e, hess_e,
                               initial_energy=0.0, final_energy=e_exc)
absorp.allow_imaginary = True
absorp.verbose = 0
res_a = absorp.run(max_quanta=8, max_modes_excited=3)

# ---- emission: the EXCITED state is `initial` ------------------------------
emiss = vibronic.FranckCondon(mol_e, hess_e, mol_g, hess_g,
                              initial_energy=e_exc, final_energy=0.0)
emiss.allow_imaginary = True
emiss.verbose = 0
res_e = emiss.run(max_quanta=8, max_modes_excited=3)

sticks_a = res_a.stick_spectrum(kind='absorption')
sticks_e = res_e.stick_spectrum(kind='emission')

print('0-0 energy  absorption %10.1f cm^-1' % units.au2wavenumber(res_a.e_00))
print('0-0 energy  emission   %10.1f cm^-1' % units.au2wavenumber(-res_e.e_00))
print('   (they must agree: the origin is a property of the state pair)')

for name, sticks in (('absorption', sticks_a), ('emission', sticks_e)):
    cm = sticks.to_unit('cm-1')
    peak = cm.energies[sticks.intensities.argmax()]
    print('\n%s: %d lines' % (name, len(cm.energies)))
    print('  range        %10.1f .. %10.1f cm^-1' % (cm.energies.min(), cm.energies.max()))
    print('  band maximum %10.1f cm^-1 = %.4f eV'
          % (peak, units.au2ev(units.wavenumber2au(peak))))

# The vertical Stokes shift, as the separation of the two broadened band maxima.
spec_a = sticks_a.to_unit('cm-1').broaden(profile='gaussian', width=400.0,
                                         unit='cm-1', npoints=6000)
spec_e = sticks_e.to_unit('cm-1').broaden(profile='gaussian', width=400.0,
                                         unit='cm-1', npoints=6000)
max_a = spec_a.x[spec_a.y.argmax()]
max_e = spec_e.x[spec_e.y.argmax()]
print('\nStokes shift from the computed band maxima: %.1f cm^-1 = %.3f eV'
      % (max_a - max_e, units.au2ev(units.wavenumber2au(max_a - max_e))))

# For comparison, the crude linear-coupling estimate 2*lambda.  It ignores the
# frequency change and the Duschinsky rotation, so it is only a rough guide --
# prefer the band maxima above.
lam2 = vibronic.analysis.stokes_shift(absorp.duschinsky)
print('linear-coupling estimate 2*lambda:          %.1f cm^-1 = %.3f eV'
      % (units.au2wavenumber(lam2), units.au2ev(lam2)))

# ---- the wrong-order request must be refused ------------------------------
try:
    res_a.stick_spectrum(kind='emission')
except ValueError as err:
    print('\nAsking for emission from the lower state correctly raises:\n  %s'
          % str(err).split('.')[0])

# ---- emission at finite temperature ---------------------------------------
# Hot bands from thermally populated vibrational levels of the emitting state.
res_hot = emiss.run(max_quanta=8, max_modes_excited=3, temperature=600.0)
hot = res_hot.init_states.sum(axis=1) > 0
print('\nat 600 K: %d of %d lines start from a vibrationally excited level,'
      % (hot.sum(), res_hot.nstate))
print('          carrying %.3f%% of the total weight'
      % (100 * (res_hot.populations * res_hot.fcf)[hot].sum()
         / (res_hot.populations * res_hot.fcf).sum()))

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('\n(matplotlib not available; skipping the plot)')
else:
    plt.plot(spec_a.x, spec_a.y / spec_a.y.max(), 'b-', label='absorption')
    plt.plot(spec_e.x, spec_e.y / spec_e.y.max(), 'r-', label='emission')
    plt.xlabel(r'photon energy / cm$^{-1}$')
    plt.ylabel('relative intensity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fc_emission.png', dpi=120)
    print('\nwrote fc_emission.png')
