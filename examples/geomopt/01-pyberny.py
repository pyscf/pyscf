#!/usr/bin/env python

'''
Use pyberny to get the molecular equilibrium geometry.
'''

from pyscf import gto, scf
from pyscf.geomopt.berny_solver import optimize

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)

#
# geometry optimization for HF
#
mol_eq = berny_solver.optimize(mf)
print(mol_eq.atom_coords())

#
# geometry optimization for CASSCF
#
from pyscf import mcscf
mf = scf.RHF(mol)
mc = mcscf.CASSCF(mf, 4, 4)
conv_params = {
    'gradientmax': 6e-3,  # Eh/AA
    'gradientrms': 2e-3,  # Eh/AA
    'stepmax': 2e-2,      # AA
    'steprms': 1.5e-2,    # AA
}
mol_eq = berny_solver.optimize(mc, **conv_params)

