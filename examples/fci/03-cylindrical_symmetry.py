#!/usr/bin/env python

'''
FCI solver direct_spin1_cyl_sym is designed for systems with cylindrical
symmetry. It is not compatible with the wavefunction obtained from
direct_spin1_symm. The wave function of direct_spin1_cyl_sym is computed with
the cylindrical symmetry adapted complex orbitals.
'''

from pyscf import gto, ao2mo
from pyscf.fci import direct_spin1_cyl_sym
import sys

verify_windows = '--pyscf-verify-windows' in sys.argv

mol = gto.Mole()
mol.build(
    atom = 'Li 0 0 0; Li 0 0 1.5',
    basis = 'sto-3g' if verify_windows else 'cc-pVDZ',
    symmetry = True,
)
mf = mol.RHF().run()
# To reduce the cost, this example takes just the first orbitals that are
# needed for the selected verification mode.
norb = 10 if verify_windows else 19
c = mf.mo_coeff[:,:norb]
orbsym = mf.mo_coeff.orbsym[:norb]

h1e = c.T.dot(mf.get_hcore()).dot(c)
eri = ao2mo.kernel(mol, c)
solver = direct_spin1_cyl_sym.FCI(mol)
symmetries = ['A1g', 'A1u', 'E1ux', 'E1uy']
if not verify_windows:
    symmetries.extend(['E1gx', 'E1gy', 'E2ux', 'E2uy', 'E2gx', 'E2gy', 'E3ux', 'E3uy', 'E3gx', 'E3gy'])
for sym in symmetries:
    e, v = solver.kernel(h1e, eri, c.shape[1], mol.nelec, orbsym=orbsym,
                         wfnsym=sym, nroots=2 if verify_windows else 5,
                         ecore=mol.energy_nuc(), verbose=0)
    print(f'{sym}: {e}')
