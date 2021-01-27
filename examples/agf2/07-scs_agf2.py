#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
An example of Spin-Component-Scaled-AGF2 calculation.
'''

from pyscf import gto, scf, agf2

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

# Run an AGF2 calculation
gf2 = agf2.AGF2(mf)
gf2.os_factor = 1.2
gf2.ss_factor = 1./3
gf2.conv_tol = 1e-7
gf2.run()

# Print the first 3 ionization potentials
gf2.ipagf2(nroots=3)

# Print the first 3 electron affinities
gf2.eaagf2(nroots=3)


# The agf2 module can also be used in this way to perform SCS-MP2 
# (c.f. 05-agf2_moments.py), or SCS-ADC(2) (c.f. 06-adc2_solver.py).
