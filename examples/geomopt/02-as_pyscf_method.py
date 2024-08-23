#!/usr/bin/env python

'''
For the customized energy and gradients function (e.g. adding DFT-D3
correction), a fake pyscf method need to be created before passing to
berny_solver.
'''

import numpy as np
from pyscf import gto, scf
from pyscf.geomopt import berny_solver, geometric_solver, as_pyscf_method

mol = gto.M(atom='N 0 0 0; N 0 0 1.8', unit='Bohr', basis='ccpvdz')
mf = scf.RHF(mol)

grad_scan = scf.RHF(mol).nuc_grad_method().as_scanner()
def f(mol):
    e, g = grad_scan(mol)
    r = mol.atom_coords()
    penalty = np.linalg.norm(r[0] - r[1])**2 * 0.1
    e += penalty
    g[0] += (r[0] - r[1]) * 2 * 0.1
    g[1] -= (r[0] - r[1]) * 2 * 0.1
    print('Customized |g|', np.linalg.norm(g))
    return e, g

#
# Function as_pyscf_method is a wrapper that convert the "energy-gradients"
# function to berny_solver.  The "energy-gradients" function takes the Mole
# object as geometry input, and returns the energy and gradients of that
# geometry.
#
fake_method = as_pyscf_method(mol, f)
new_mol = berny_solver.optimize(fake_method)

print('Old geometry:')
print(mol.tostring())

print('New geometry:')
print(new_mol.tostring())

#
# Geometry can be also optimized with geomeTRIC library
#
new_mol = geometric_solver.optimize(fake_method)

