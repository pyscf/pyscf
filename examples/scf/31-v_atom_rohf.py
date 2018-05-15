#!/usr/bin/env python

'''
Spherical symmetry needs to be carefully treated in the atomic calculation.
The default initial guess may break the spherical symmetry.  To preserve the
spherical symmetry in the atomic calculation, it is often needed to tune the
initial guess and SCF model.

See also 31-cr_atom_rohf_tune_init_guess.py
'''

import numpy
from pyscf import gto, scf

#
# Method 1: Construct the atomic initial guess from cation.
#
mol = gto.Mole()
mol.verbose = 4
mol.atom = 'V'
mol.basis = 'ccpvtz'
mol.symmetry = True
mol.spin = 0
mol.charge = 5
mol.build()

mf = scf.ROHF(mol)
irrep_nelec = {'A1g' :(4,3), 'E1gx': (1,0), 'E1gy': (1,0), 'E2gx': (1,0), 'E2gy': (1,0),
               'A1u' :4, 'E1ux': 4, 'E1uy': 4}
mf.kernel()
# The output of .analyze() method can help to identify whether the spherical
# symmetry is conserved.
#mf.analyze()

# Set the system back to neutral atom
mol.spin = 5
mol.charge = 0

mf.irrep_nelec = mf.get_irrep_nelec()
mf.irrep_nelec['A1g'] = (4,3)
mf.irrep_nelec['E1gx'] = (1,0)
mf.irrep_nelec['E1gy'] = (1,0)
mf.irrep_nelec['E2gx'] = (1,0)
mf.irrep_nelec['E2gy'] = (1,0)
dm = mf.make_rdm1()
mf.kernel(dm)
#mf.analyze()

#
# Regular SCF iteration may break the spherical symmetry in may systems.
# Second order SCF model often works slightly better.
#
mf = mf.newton()
mf.kernel(dm)
#mf.analyze()


#
# Method 2: Construct the atomic initial guess of a large basis set from a
# calculation of small basis set.
#
mol = gto.Mole()
mol.verbose = 4
mol.atom = 'V'
mol.basis = 'minao'
mol.symmetry = True
mol.spin = 0
mol.charge = 5
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

#
# Setup the system with large basis set
#
mol = gto.Mole()
mol.verbose = 4
mol.atom = 'V'
mol.basis = 'ccpvtz'
mol.symmetry = True
mol.spin = 5
mol.charge = 0
mol.build()

dm = mf.make_rdm1()
dm = scf.addons.project_dm_nr2nr(mol, dm, mol1)
mf = scf.ROHF(mol1)
mf.kernel(dm)
#mf.analyze()

#
# Second order SCF can be applied on the project density matrix as well
#
mf = mf.newton()
mf.kernel(dm)
#mf.analyze()

