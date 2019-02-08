#!/usr/bin/env python

'''
Use geomeTRIC library to optimize the molecular geometry.
'''

from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)

#
# geometry optimization for HF
#
mol_eq = optimize(mf)
print(mol_eq.atom_coords())

#
# geometry optimization for CASSCF
#
from pyscf import mcscf
mf = scf.RHF(mol)
mc = mcscf.CASSCF(mf, 4, 4)
mol_eq = optimize(mc)

