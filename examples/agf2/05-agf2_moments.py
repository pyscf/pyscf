#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
An example of higher moment self-consistency in renormalization of self-energy in AGF2.

Note that only the default AGF2 is efficiently implemented (corresponding to AGF2(1,0) in 
the literature, and equivalent to AGF2(None,0)). Higher moments do not support density-fitting 
or parallelism.

Default AGF2 corresponds to the AGF2(1,0) method outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 1090 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., 16, 6294 (2020).
'''

from __future__ import print_function
from pyscf import gto, scf, agf2, mp

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='6-31g')

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

# Get the canonical MP2
mp2 = mp.MP2(mf)
mp2.run()

# We can use very high moment calculations to converge to traditional GF2 limit.
# We can also use the zeroth iteration to quantify the AGF2 error by comparison
# to the MP2 energy.
gf2_56 = agf2.AGF2(mf, nmom=(5,6))
gf2_56.build_se()
e_mp2 = gf2_56.energy_mp2()

print('Canonical MP2 Ecorr: ', mp2.e_corr)
print('AGF2(%s,%s) MP2 Ecorr: '%gf2_56.nmom, e_mp2)
print('Error: ', abs(mp2.e_corr - e_mp2))

# Run a high moment AGF2(5,6) calculation and compare to AGF2 (AGF2 as
# default in pyscf is technically AGF2(None,0)). See
# second reference for more details.
gf2_56.run()

gf2 = agf2.AGF2(mf)
gf2.run()

print('E(corr):')
print('AGF2(%s,%s): '%gf2_56.nmom, gf2_56.e_corr)
print('AGF2(1,0): ', gf2.e_corr)

print('IP:')
print('AGF2(%s,%s): '%gf2_56.nmom, gf2_56.ipagf2(nroots=1)[0])
print('AGF2(1,0): ', gf2.ipagf2(nroots=1)[0])
