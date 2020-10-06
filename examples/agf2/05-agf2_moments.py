#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#

'''
An example of AGF2 for higher moment consistency
'''

from __future__ import print_function
from pyscf import gto, scf, agf2, mp

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='6-31g', verbose=0)

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

# Get the canonical MP2
mp2 = mp.MP2(mf)
mp2.run()

# We can use very high moment calculations to approximate GF2 - this
# is computationally unfeasible but we can use the zeroth iteration
# to verify that we match the MP2 energy.
gf2 = agf2.AGF2(mf, nmom=(8,8))
gf2.build_se()
e_mp2 = gf2.energy_mp2()

print('Canonical MP2: ', mp2.e_corr)
print('AGF2(%s,%s) MP2: '%gf2.nmom, e_mp2)
print('Error: ', abs(mp2.e_corr - e_mp2))

# Run an AGF2(3,4) calculation and compare to AGF2 (AGF2 as
# implemented in pyscf is technically AGF2(None,0)):
gf2_3_4 = agf2.AGF2(mf, nmom=(3,4))
gf2_3_4.run()

gf2 = agf2.AGF2(mf)
gf2.run()

print('E(corr):')
print('AGF2(%s,%s): '%gf2_3_4.nmom, gf2_3_4.e_corr)
print('AGF2: ', gf2.e_corr)

print('IP:')
print('AGF2(%s,%s): '%gf2_3_4.nmom, gf2_3_4.ipagf2(nroots=1)[0])
print('AGF2: ', gf2.ipagf2(nroots=1)[0])
