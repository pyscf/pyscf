#!/usr/bin/env python

'''
Localized orbitals can be obtained via pivoted Cholesky factorization
of the density matrix. The procedure is iteration-free and produces
unique orbitals (except for degeneracies etc.), albeit they are
less well localized compared with other procedures.

F. Aquilante, T.B. Pedersen, J. Chem. Phys. 125, 174101 (2006)
https://doi.org/10.1063/1.2360264

This example performs separate (split) localizations of
1) the occupied orbitals and
2) the virtual orbitals,
and produces a molden output containing both orbital sets.
'''

import numpy
from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.lo import cholesky_mos
from pyscf.tools import molden

# Set up of the molecule (C3H7OH)
mol = Mole()
mol.atom = '''
C        0.681068338      0.605116159      0.307300799
C       -0.733665805      0.654940451     -0.299036438
C       -1.523996730     -0.592207689      0.138683275
H        0.609941801      0.564304456      1.384183068
H        1.228991034      1.489024155      0.015946420
H       -1.242251083      1.542928348      0.046243898
H       -0.662968178      0.676527364     -1.376503770
H       -0.838473936     -1.344174292      0.500629028
H       -2.075136399     -0.983173387     -0.703807608
H       -2.212637905     -0.323898759      0.926200671
O        1.368219958     -0.565620846     -0.173113101
H        2.250134219     -0.596689848      0.204857736
'''
mol.basis = 'def2-SVP'
mol.build()

# Perform a Hartree-Fock calculation
mf = RHF(mol)
mf.kernel()

# determine the number of occupied orbitals
nocc = numpy.count_nonzero(mf.mo_occ > 0)
# localize the occupied orbitals separately
lmo_occ = cholesky_mos(mf.mo_coeff[:, :nocc])
# localize the virtual orbitals separately
lmo_virt = cholesky_mos(mf.mo_coeff[:, nocc:])
# merge the MO coefficients in one matrix
lmo_merged = numpy.hstack((lmo_occ, lmo_virt))
# dump the merged MO coefficients in a molden file
filename = 'c3h7oh_cholesky.molden'
print('Dumping the orbitals in file:', filename)
molden.from_mo(mol, filename, lmo_merged, occ=mf.mo_occ)
