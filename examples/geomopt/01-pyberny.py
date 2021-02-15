#!/usr/bin/env python

'''
Use pyberny to get the molecular equilibrium geometry.
'''

from pyscf import gto, scf
from pyscf.geomopt.berny_solver import optimize

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)

#
# geometry optimization for HF.  There are two entries to invoke the berny
# geometry optimization.
#
# method 1: import the optimize function from pyscf.geomopt.berny_solver
mol_eq = optimize(mf)
print(mol_eq.atom_coords())

# method 2: create the optimizer from Gradients class
mol_eq = mf.Gradients().optimizer(solver='berny').kernel()

#
# geometry optimization for CASSCF
#
from pyscf import mcscf
mf = scf.RHF(mol)
mc = mcscf.CASSCF(mf, 4, 4)
conv_params = {
    'gradientmax': 6e-3,  # Eh/Bohr
    'gradientrms': 2e-3,  # Eh/Bohr
    'stepmax': 2e-2,      # Bohr
    'steprms': 1.5e-2,    # Bohr
}
# method 1
mol_eq = optimize(mc, **conv_params)

# method 2
mol_eq = mc.Gradients().optimizer(solver='berny').kernel(conv_params)
