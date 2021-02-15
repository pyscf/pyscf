#!/usr/bin/python

'''
This example shows how to diagonalize the density matrix using function
symm.eigh to get the symmetry adapted natural orbitals.
'''

import numpy
import numpy.linalg
from pyscf import gto, scf, mp, symm

mol = gto.M(atom='C 0 0 0; C 0 0 1.24253', basis='cc-pvdz', symmetry=1,
            symmetry_subgroup='D2h')

# run HF and store HF results in mf
mf = scf.RHF(mol).run()

# run MP2 and store MP2 results in mp2
mp2 = mp.MP2(mf).run()

rdm1 = numpy.diag(mf.mo_occ)  # The HF 1-particle density matrix in MO representation
rdm1 += mp2.make_rdm1()  # Add the correlation part

natocc, natorb = numpy.linalg.eigh(rdm1)
# Note natorb is in MO basis representation
natorb = numpy.dot(mf.mo_coeff, natorb)
try:
    natorb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, natorb)
    print(natorb_sym)
except ValueError:
    print('The diagonalization for rdm1 breaks the symmetry of the degenerated natural orbitals.')

print('\nIn eigenvalue sovler symm.eigh, we diagonalize the density matrix with the MO symmetry '
      'information.  The eigenvectors are all symmetry adapted.')
orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff)
natocc, natorb = symm.eigh(rdm1, orbsym)
natorb = numpy.dot(mf.mo_coeff, natorb)
natorb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, natorb)
print(natorb_sym)

