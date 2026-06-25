#!/usr/bin/env python

'''
Geometry optimization with ASE

ASE uses the BFGS optimizer by default. This general-purpose algorithm is often
not as efficient as dedicated geometry optimizers such as Berny or geomeTRIC.
'''

import pyscf
import sys

verify_windows = '--pyscf-verify-windows' in sys.argv

mol = pyscf.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
try:
    mf_opt = mol.RHF().Gradients().optimizer(solver='ase')
except ModuleNotFoundError:
    if verify_windows:
        # ASE is an optional dependency for this geometry optimizer example.
        print('Skipping ASE geometry optimization example during Windows verification because ase is not installed.')
        raise SystemExit(0)
    raise

mol_eq = mf_opt.kernel()
print(mol_eq.tostring())
