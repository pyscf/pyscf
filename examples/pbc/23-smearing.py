#!/usr/bin/env python

'''Fermi-Dirac or Gaussian smearing for PBC SCF calculations

In metallic systems, the small gaps can lead to slow SCF convergence.
The smearing technique can aid SCF convergence by introducing fractional
occupancy near the Fermi level.
'''

import pyscf

cell = pyscf.M(
a='''
0.   2.02 2.02
2.02 0.   2.02
2.02 2.02 0.
''',
atom='Al 0 0 0',
basis='gth-dzvp',
pseudo='gth-pbe',
verbose=5)

kmesh = [4,4,4]
mf = cell.KRKS(xc='pbe', kpts=cell.make_kpts(kmesh))
#
# mf.smearing() method creates a new SCF object with smearing enabled.
#
# sigma : smearing width (Hartree)
# method: 'fermi' for Fermi-Dirac smearing, 'gauss' for Gaussian smearing
#
mf = mf.smearing(sigma=0.1, method='fermi')
mf.kernel()
print('Entropy = %s' % mf.entropy)
print('Free energy = %s' % mf.e_free)
print('Approximate zero temperature energy = %s' % ((mf.e_tot+mf.e_free)/2))

# Smearing parameters can be adjusted at runtime to help SCF convergence.
# The converged orbitals and density from the previous SCF run are reused
# as the initial guess for the next calculation.
#
# For example, the smearing width (sigma) can be reduced gradually
# (e.g., 0.1 -> 0.01 -> 0.001) to approach the zero-temperature limit.
#
# Note, very small sigma (e.g., 1e-4 or smaller) may result in numerical
# instability. Setting sigma = 0 disables smearing.

mf.sigma = .01
mf.kernel()

mf.sigma = 0.001
mf.kernel()
