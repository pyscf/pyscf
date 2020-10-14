#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#

'''
An example of AGF2 within the frozen core approximation.

AGF2 corresponds to the AGF2(None,0) method outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 2 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., 16, 6294 (2020).
'''

from pyscf import gto, scf, agf2, mp

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

# Run an AGF2 calculation
gf2 = agf2.AGF2(mf)
gf2.conv_tol = 1e-7
gf2.frozen = 2
gf2.run()

# Print the first 3 ionization potentials
gf2.ipagf2(nroots=3)

# Print the first 3 electron affinities
gf2.eaagf2(nroots=3)


# Check that a high-moment calculation is equal to MP2 in the first iteration
mol = gto.M(atom='H 0 0 0; Li 0 0 1', basis='cc-pvdz')

mf = scf.RHF(mol)
mf.run()

mp2 = mp.MP2(mf)
mp2.frozen = 1
mp2.run()

gf2 = agf2.AGF2(mf, nmom=(5,6))
gf2.frozen = mp2.frozen
gf2.run(max_cycle=1)

print(mp2.e_corr)
print(gf2.e_init)
