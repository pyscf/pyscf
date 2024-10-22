#!/usr/bin/env python

'''
Use geomeTRIC library to optimize the molecular geometry.
'''

from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)

#
# geometry optimization for HF.  There are two entries to invoke the geomeTRIC
# optimization
#
# method 1: import the optimize function from pyscf.geomopt.geometric_solver
mol_eq = optimize(mf)
print(mol_eq.tostring())
print('Atomic coordinates (Ang):')
print(mol_eq.atom_coords(unit='Ang'))
print('Atomic coordinates (Bohr):')
print(mol_eq.atom_coords(unit='Bohr'))

# method 2: create the optimizer from Gradients class
mol_eq = mf.Gradients().optimizer(solver='geomeTRIC').kernel()

#
# geometry optimization for CASSCF
#
from pyscf import mcscf
mf = scf.RHF(mol)
mc = mcscf.CASSCF(mf, 4, 4)
conv_params = {
    'convergence_energy': 1e-4,  # Eh
    'convergence_grms': 3e-3,    # Eh/Bohr
    'convergence_gmax': 4.5e-3,  # Eh/Bohr
    'convergence_drms': 1.2e-2,  # Angstrom
    'convergence_dmax': 1.8e-2,  # Angstrom
}
# method 1
mol_eq = optimize(mc, **conv_params)

# method 2
mol_eq = mc.Gradients().optimizer(solver='geomeTRIC').kernel(conv_params)


#
# geometry optimization for DFT, MP2, CCSD
#
mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
            ''', basis='3-21g')

mf = mol.RKS(xc='pbe,')
mol1 = optimize(mf)

mymp2 = mol.MP2()
mol1 = optimize(mymp2)

mycc = mol.CCSD()
mol1 = optimize(mycc)
