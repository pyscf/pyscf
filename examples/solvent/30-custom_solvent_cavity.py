#!/usr/bin/env python

'''
Custom solvent cavity
'''

import numpy
from pyscf import gto, qmmm, solvent

#
# Case 1. Cavity for dummy atoms with basis on the dummy atoms
#
mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
X-C      0.000000    0.000000             -1.5
X-O      0.000000    0.000000              1.6
            ''',
            verbose = 4)

sol = solvent.ddCOSMO(mol)
cavity_radii = sol.get_atomic_radii()

cavity_radii[4] = 3.0  # Bohr, for X-C
cavity_radii[5] = 2.5  # Bohr, for X-O
# Overwrite the get_atom_radii method to feed the custom cavity into the solvent model
sol.get_atomic_radii = lambda: cavity_radii

mf = mol.RHF().ddCOSMO(sol)
mf.run()


#
# Case 2. Cavity for dummy atoms (without basis)
#
mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)

# Use a MM molecule to define cavity from dummy atoms.
# See also the example 22-with_qmmm.py
coords = numpy.array([
    [0, 0, -1.5],
    [0, 0, 1.6],
])
charges = numpy.array([0, 0])
mm_atoms = [('X', c) for c in coords]
mm_mol = qmmm.create_mm_mol(mm_atoms, charges)

# Make a giant system include both QM and MM particles
qmmm_mol = mol + mm_mol

# The solvent model is based on the giant system
sol = solvent.ddCOSMO(qmmm_mol)
cavity_radii = sol.get_atomic_radii()

# Custom cavity
cavity_radii[4] = 3.0  # Bohr
cavity_radii[5] = 2.5  # Bohr
# Overwrite the get_atom_radii method to feed the custom cavity into the solvent model
sol.get_atomic_radii = lambda: cavity_radii

mf = mol.RHF().QMMM(coords, charges)
mf = mf.ddCOSMO(sol)
mf.run()
