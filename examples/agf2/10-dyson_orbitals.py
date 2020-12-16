#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
An example of generating Dyson orbitals from an AGF2 calculation.

Default AGF2 corresponds to the AGF2(1,0) method outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 1090 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., 16, 6294 (2020).
'''

from pyscf import gto, scf, agf2

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

# Run an AGF2 calculation
gf2 = agf2.AGF2(mf)
gf2.conv_tol = 1e-7
gf2.run(verbose=4)

# Access the Dyson orbitals (vectors in AO basis):
dyson_orbitals = gf2.qmo_coeff

# Find the Dyson orbital corresponding to the HOMO and LUMO:
dyson_homo = gf2.qmo_coeff[:,gf2.qmo_occ > 0][:,-1]
dyson_lumo = gf2.qmo_coeff[:,gf2.qmo_occ == 0][:,0]
