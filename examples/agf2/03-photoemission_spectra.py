#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#

'''
Use the converged Green's function to build a photoemission spectrum
following an AGF2 calculation.
'''

import numpy
from pyscf import gto, scf, agf2

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

# Run an AGF2 calculation
gf2 = agf2.AGF2(mf)
gf2.conv_tol = 1e-7
gf2.run()

# Access the GreensFunction object and compute the spectrum
gf = gf2.gf
grid = numpy.arange(-10.0, 10.0, 1000)
eta = 0.02
spectrum = gf.real_freq_spectrum(grid, eta=eta)

# The array `spectrum` is now a (nfreq x nmo x nmo) array of the
# spectral function -1/pi * G(\omega + i\eta).
