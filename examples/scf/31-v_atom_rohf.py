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
mf.kernel()
# The output of .analyze() method can help to verify whether the spherical
# symmetry is conserved.
#mf.analyze()

# Restore the neutral atom
mol.spin = 5
mol.charge = 0

mf.irrep_nelec = mf.get_irrep_nelec()
mf.irrep_nelec['s+0'] = (3,3)
mf.irrep_nelec['d-2'] = (1,0)
mf.irrep_nelec['d-1'] = (1,0)
mf.irrep_nelec['d+0'] = (1,0)
mf.irrep_nelec['d+1'] = (1,0)
mf.irrep_nelec['d+2'] = (1,0)
dm = mf.make_rdm1()
mf.kernel(dm)
#mf.analyze()

#
# Regular SCF iterations sometimes break the spherical symmetry while the
# second order SCF method works slightly better.
#
mf = mf.newton()
mf.kernel(dm)
#mf.analyze()


#
# Method 2: Construct the atomic initial guess of large basis from a
# calculation of small basis.
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
mol1 = gto.Mole()
mol1.verbose = 4
mol1.atom = 'V'
mol1.basis = 'ccpvtz'
mol1.symmetry = True
mol1.spin = 5
mol1.charge = 0
mol1.build()

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


########################################
# Spherical symmetry was not supported until PySCF-1.7.4. SO3 symmetry was
# recogonized as Dooh. Code below is token from old examples.
#
# Construct the atomic initial guess from cation.
#
mol = gto.Mole()
mol.verbose = 4
mol.atom = 'V'
mol.basis = 'ccpvtz'
mol.symmetry = 'Dooh'
mol.spin = 0
mol.charge = 5
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

# Restore the neutral atom
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
