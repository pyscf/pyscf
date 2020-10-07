#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#

'''
A simple example of unrestricted AGF2.

AGF2 corresponds to the AGF2(None,0) method outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 2 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., X, X (2020).
'''

from pyscf import gto, scf, agf2

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', charge=-1, spin=1)

mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.run()

# Run an AGF2 calculation
gf2 = agf2.AGF2(mf)
gf2.conv_tol = 1e-7
gf2.run(verbose=4)

# Print the first 3 ionization potentials
gf2.ipagf2(nroots=3)

# Print the first 3 electron affinities
gf2.eaagf2(nroots=3)
