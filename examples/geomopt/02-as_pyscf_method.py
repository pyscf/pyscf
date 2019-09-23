#!/usr/bin/env python

'''
For the customized energy and gradients function (e.g. adding DFT-D3
correction), a fake pyscf method need to be created before passing to
berny_solver.
'''

import numpy as np
from pyscf import gto, scf
from pyscf.geomopt import berny_solver

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

fake_method = berny_solver.as_pyscf_method(mol, f)
mol1 = berny_solver.optimize(fake_method)
