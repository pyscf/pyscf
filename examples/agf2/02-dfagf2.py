#!/usr/bin/env python
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
An example of restricted AGF2 with density fitting, obtaining the 1RDM and dipole moment

Default AGF2 corresponds to the AGF2(1,0) method outlined in the papers:
  - O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 1090 (2020).
  - O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., 16, 6294 (2020).
'''

from pyscf import gto, scf, agf2
import numpy as np
from functools import reduce

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

mf = scf.RHF(mol).density_fit(auxbasis='cc-pv5z-ri')
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

# Get the MO-basis density matrix and calculate dipole moments:
dm = gf2.make_rdm1()

dipole = [0.0, 0.0, 0.0]
# Transform dipole moment integrals into MO basis
mol.set_common_origin([0,0,0])
r_ints_ao = mol.intor('cint1e_r_sph', comp=3)
r_ints_mo = np.empty_like(r_ints_ao)
for i in range(3):
    r_ints_mo[i] = reduce(np.dot,(mf.mo_coeff.T, r_ints_ao[i], mf.mo_coeff))
    dipole[i] = -np.trace(np.dot(dm, r_ints_mo[i]))
    # Add nuclear component
    for j in range(mol.natm):
        dipole[i] += mol.atom_charge(j) * mol.atom_coord(j)[i]

print('Dipole moment from AGF2: {} {} {}'.format(dipole[0], dipole[1], dipole[2]))
