#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#

'''
Use the converged Green's function to build a photoemission spectrum
following an AGF2 calculation.

AGF2 corresponds to the AGF2(None,0) method outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 2 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., X, X (2020).
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
grid = numpy.linspace(-10.0, 10.0, 1000)
eta = 0.02
spectrum = gf.real_freq_spectrum(grid, eta=eta)

# The array `spectrum` is now a length nfreq array of the
# spectral function -1/pi * Tr[Im[G(\omega + i\eta)]].

# We can also build the self-energy on the real-frequency axis
# by accessing the poles:
e = gf2.se.energy - gf2.se.chempot
v = gf2.se.coupling
denom = grid[:,None] - (e + numpy.sign(e)*eta*1.0j)[None]
se = numpy.einsum('xk,yk,wk->wxy', v, v.conj(), 1./denom)
