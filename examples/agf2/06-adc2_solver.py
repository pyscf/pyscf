#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#

'''
An example of the connection between AGF2 and ADC(2)

AGF2 in the zeroth iteration with no compression is equivalent
to ADC(2). AGF2 is not recommended as a practical solver for
ADC(2), this is mostly to demonstrate the link between the
methods.

AGF2 corresponds to the AGF2(None,0) method outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 2 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., X, X (2020).
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

# Use the adc module to get the 1p space from ADC(2):
adc2 = adc.radc.RADCIP(adc.ADC(mf).run())
h_1p = adc2.get_imds()

# Find the eigenvalues of the self-energy in the 'extended Fock matrix'
# format, which are the ionization potentials:
e_ip = se.eig(h_1p)[0]
print('IP-ADC(2) using the AGF2 solver:')
print(-e_ip[-1])

# Compare to ADC(2) values - note that these will not be the same, since
# AGF2 performs a compression to allow full diagonalisation of the 
# Hamiltonian instead of using an iterative solver:
e_ip = adc2.kernel(nroots=1)[0]
print('IP-ADC(2) using the pyscf solver:')
print(e_ip[-1])

# One may calculate the uncompressed self-energy using RAGF2_slow, and
# then use the Davidson solver in pyscf.agf2.aux to get the full ADC(2)
# values:
gf2 = agf2.AGF2(mf, nmom=(None, None))
se = gf2.build_se()
se = se.get_occupied() # 2p1h/1p2h -> 2p1h
se.coupling = se.coupling[:gf2.nocc] # 1p/1h -> 1p
e_ip = agf2.aux.davidson(se, h_1p, nroots=1)[1]
print('IP-ADC(2) using the AGF2_slow solver:')
print(-e_ip[-1])
