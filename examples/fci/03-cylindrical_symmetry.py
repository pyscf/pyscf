#!/usr/bin/env python

'''
FCI solver direct_spin1_cyl_sym is designed for systems with cylindrical
symmetry. It is not compatible with the wavefunction obtained from
direct_spin1_symm. The wave function of direct_spin1_cyl_sym is computed with
the cylindrical symmetry adapted complex orbitals.
'''

from pyscf import gto, ao2mo
import numpy
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
energies = {}
for sym in ['A1g', 'A1u', 'E1ux', 'E1uy', 'E1gx', 'E1gy', 'E2ux', 'E2uy',
            'E2gx', 'E2gy', 'E3ux', 'E3uy', 'E3gx', 'E3gy']:
    energies[sym] = numpy.asarray(solver.kernel(
        h1e, eri, c.shape[1], mol.nelec, orbsym=orbsym, wfnsym=sym,
        nroots=5, ecore=mol.energy_nuc(), verbose=0)[0])
    print(f'{sym}: {energies[sym]}')

# The x and y components of each E irrep are degenerate, and irreps of
# different Lambda do not alias each other (issue #1517: in the D2h subgroup
# E2gx was indistinguishable from A1g and E3ux from E1ux)
for w in ['E1u', 'E1g', 'E2u', 'E2g', 'E3u', 'E3g']:
    assert max(abs(energies[w+'x'] - energies[w+'y'])) < 1e-7
for a, b in [('A1g', 'E2g'), ('A1u', 'E2u'), ('E1u', 'E3u'), ('E1g', 'E3g')]:
    ka = a if a[0] == 'A' else a + 'x'
    kb = b if b[0] == 'A' else b + 'x'
    assert max(abs(energies[ka] - energies[kb])) > 1e-3
