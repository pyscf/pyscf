#!/usr/bin/env python

'''
Automatically apply SOSCF for unconverged SCF calculations in geometry optimization.
'''

import pyscf
from pyscf.geomopt import geometric_solver, as_pyscf_method

# Force SCF to stop early at an unconverged state
pyscf.scf.hf.RHF.max_cycle = 5

mol = pyscf.M(
    atom='''
    O 0 0 0
    H 0 .7 .8
    H 0 -.7 .8
    ''')
mf = mol.RHF()
try:
    mol_eq = geometric_solver.optimize(mf)
except RuntimeError:
    print('geometry optimization should stop for unconverged calculation')

# Apply SOSCF when SCF is not converged
mf_scanner = mf.as_scanner()
def apply_soscf_as_needed(mol):
    mf_scanner(mol)
    if not mf_scanner.converged:
        mf_soscf = mf_scanner.newton().run()
        for key in ['converged', 'mo_energy', 'mo_coeff', 'mo_occ', 'e_tot', 'scf_summary']:
            setattr(mf_scanner, key, getattr(mf_soscf, key))
    grad = mf_scanner.Gradients().kernel()
    return mf_scanner.e_tot, grad

mol_eq = geometric_solver.optimize(as_pyscf_method(mol, apply_soscf_as_needed))
