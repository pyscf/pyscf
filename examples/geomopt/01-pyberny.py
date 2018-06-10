#!/usr/bin/env python

'''
Use pyberny to get the molecular equilibrium geometry.
'''

from pyscf import gto, scf
from pyscf.geomopt import berny_solver

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)

mol_eq = berny_solver.optimize(mf)
print(mol_eq.atom_coords())
