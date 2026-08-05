#!/usr/bin/env python

'''
Harmonic normal modes in atomic units, for Franck-Condon work.

pyscf.vibronic.HarmonicModel is the input object of every Franck-Condon
calculation.  It differs from pyscf.hessian.thermo.harmonic_analysis in two ways
that matter here:

  * it works in true atomic units (masses in electron masses), so the angular
    frequency `omega` is directly an energy in Hartree with hbar = 1;
  * it keeps the mass-weighted mode matrix L, which the Duschinsky
    transformation needs and which thermo discards.

The resulting wavenumbers are identical to thermo's, as this example shows.
'''

import numpy
from pyscf import gto, scf, vibronic
from pyscf.vibronic import units
from pyscf.hessian import thermo

mol = gto.M(
    atom = '''O    0.   0.       0.1173
              H    0.   0.7572  -0.4692
              H    0.  -0.7572  -0.4692''',
    basis = '631g')

mf = mol.RHF().run()
hessian = mf.Hessian().kernel()

# The Hessian may be passed either in the (natm,natm,3,3) layout returned by
# mf.Hessian().kernel() or flattened to (3N,3N); the shape is detected.
model = vibronic.HarmonicModel.from_mole(mol, hessian, energy=mf.e_tot)

print('rotor type          %s' % model.rotor_type)
print('vibrational modes   %d  (3N-6 = %d)' % (model.nvib, 3 * mol.natm - 6))
print('frequencies [cm^-1] %s' % numpy.round(model.freq_wavenumber, 2))
print('omega [Eh]          %s' % numpy.round(model.freq, 6))
print('zero-point energy   %.8f Eh = %.2f cm^-1' % (model.zpe,
                                                    units.au2wavenumber(model.zpe)))

# The mode matrix L is (3N, nvib) with orthonormal columns, in mass-weighted
# Cartesian coordinates.  Q = L^T M^(1/2) (x - x0) is the normal coordinate.
print('\nL shape             %s' % (model.modes.shape,))
print('max|L^T L - I|      %.2e' % abs(model.modes.T.dot(model.modes)
                                       - numpy.eye(model.nvib)).max())

# Cross-check against the existing thermochemistry module.
ref = thermo.harmonic_analysis(mol, hessian)
print('max rel. difference vs hessian.thermo: %.2e'
      % abs(model.freq_wavenumber / ref['freq_wavenumber'].real - 1).max())

# Isotopic substitution: pass masses in amu.  Only the nuclear masses change,
# not the electronic structure, so the same geometry and Hessian are reused.
d2o = vibronic.HarmonicModel.from_mole(mol, hessian, energy=mf.e_tot,
                                       mass=[15.9949, 2.0141, 2.0141])
print('\nH2O [cm^-1] %s' % numpy.round(model.freq_wavenumber, 1))
print('D2O [cm^-1] %s' % numpy.round(d2o.freq_wavenumber, 1))
print('ratio       %s' % numpy.round(model.freq_wavenumber / d2o.freq_wavenumber, 4))

# A detailed listing of the modes.
model.dump_normal_modes()
