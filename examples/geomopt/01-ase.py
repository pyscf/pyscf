#!/usr/bin/env python

'''
Geometry optimization with ASE

ASE uses the BFGS optimizer by default. This general-purpose algorithm is often
not as efficient as dedicated geometry optimizers such as Berny or geomeTRIC.
'''

import pyscf

mol = pyscf.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf_opt = mol.RHF().Gradients().optimizer(solver='ase')

mol_eq = mf_opt.kernel()
print(mol_eq.tostring())
