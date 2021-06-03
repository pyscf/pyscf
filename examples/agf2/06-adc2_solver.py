#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
An example of the connection between AGF2 and ADC(2)

AGF2 without self-consistency, with no renormalization of the auxiliary space, 
and with occupied/virtual separation, is equivalent to ADC(2). AGF2 is not 
recommended as a practical solver for ADC(2), but this example numerically 
demonstrates this correspondance between the methods.

The AGF2 method is outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 1090 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., 16, 6294 (2020).
'''

import numpy
from pyscf import gto, scf, agf2, adc

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

# We can build the compressed ADC(2) self-energy as the zeroth iteration
# of AGF2, taking only the space corresponding to 1p coupling to 2p1h
gf2 = agf2.AGF2(mf)
se = gf2.build_se()
se = se.get_occupied() # 2p1h/1p2h -> 2p1h
se.coupling = se.coupling[:gf2.nocc] # 1p/1h -> 1p

# Use the adc module to get the 1p space from ADC(2). In AGF2, this is the
# bare Fock matrix, and is relaxed through the self-consistency. We can use
# the ADC 1p space instead.
adc2 = adc.radc.RADCIP(adc.ADC(mf).run())
h_1p = adc2.get_imds()

# Find the eigenvalues of the self-energy in the 'extended Fock matrix'
# format, which are the ionization potentials:
e_ip = se.eig(h_1p)[0]
print('IP-ADC(2) using the AGF2 solver (with renormalization):')
print(-e_ip[-1])

# Compare to ADC(2) values - note that these will not be the same, since
# AGF2 performs a compression/renormalization of the 2p1h space to allow 
# full diagonalisation of the Hamiltonian instead of using an iterative solver:
e_ip = adc2.kernel(nroots=1)[0]
print('IP-ADC(2) using the pyscf solver:')
print(e_ip[-1])

# One may instead calculate the uncompressed self-energy (slow code), and
# then use the Davidson solver in pyscf.agf2.aux to get the exact ADC(2)
# values:
gf2 = agf2.AGF2(mf, nmom=(None, None))
se = gf2.build_se()
se = se.get_occupied() # 2p1h/1p2h -> 2p1h
se.coupling = se.coupling[:gf2.nocc] # 1p/1h -> 1p
e_ip = agf2.aux.davidson(se, h_1p, nroots=1)[1]
print('IP-ADC(2) via AGF2 without self-consistency or auxiliary renormalization:')
print(-e_ip[-1])
