#!/usr/bin/env python

'''
FCI solver direct_spin1_cyl_sym is designed for systems with cylindrical
symmetry. It is not compatible with the wavefunction obtained from
direct_spin1_symm. The wave function of direct_spin1_cyl_sym is computed with
the cylindrical symmetry adapted complex orbitals.
'''

from pyscf import gto, ao2mo
from pyscf.fci import direct_spin1_cyl_sym

mol = gto.Mole()
mol.build(
    atom = 'Li 0 0 0; Li 0 0 1.5',
    basis = 'cc-pVDZ',
    symmetry = True,
)
mf = mol.RHF().run()
# To reduce the cost, this example takes just the first 19 orbitals
c = mf.mo_coeff[:,:19]
orbsym = mf.mo_coeff.orbsym[:19]

h1e = c.T.dot(mf.get_hcore()).dot(c)
eri = ao2mo.kernel(mol, c)
solver = direct_spin1_cyl_sym.FCI(mol)
for sym in ['A1g', 'A1u', 'E1ux', 'E1uy', 'E1gx', 'E1gy', 'E2ux', 'E2uy', 'E2gx', 'E2gy', 'E3ux', 'E3uy', 'E3gx', 'E3gy']:
    e, v = solver.kernel(h1e, eri, c.shape[1], mol.nelec, orbsym=orbsym,
                         wfnsym=sym, nroots=5, ecore=mol.energy_nuc(), verbose=0)
    print(f'{sym}: {e}')
