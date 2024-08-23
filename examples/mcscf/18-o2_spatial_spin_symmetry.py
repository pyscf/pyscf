#!/usr/bin/env python

'''
A small O2 CASSCF calculation, in which spin and spatial symmetry should be
carefully chosen.
'''

from pyscf import gto, scf, mcscf, fci

mol = gto.M(atom='O; O 1 1.2', basis='ccpvdz', spin=2, symmetry=1)
mf = scf.RHF(mol).run()
norb = 4
nelec = 6 # leads to 4 alpha electrons and 2 beta electons due to spin=2 (Sz=1)
mc = mcscf.CASSCF(mf, norb, nelec)
try:
    mc.kernel()
except RuntimeError as e:
    print('When symmetry is enabled, FCI solver optimize the wfn of A1g symmetry by default.')
    print('In this CAS space, wfn of A1g (Sigma^+_g) symmetry for Sz=1 does not exist.')
    print(e)

# Target the ground state is A2g (Sigma^-_g) with Sz=0.
# Specify 3 alpha electrons and 3 beta electrons for CASSCF calculation.
# The Sz value for CASCI/CASSCF does not have to be the same to the mol.spin.
nelec = (3, 3)
mc = mcscf.CASSCF(mf, norb, nelec)
mc.fcisolver.wfnsym = 'A2g'
mc.kernel()
#
# Print out the largest CI coefficients
#
for c, deta, detb in fci.addons.large_ci(mc.ci, norb, nelec, tol=.01, return_strs=False):
    print(deta.tolist(), detb.tolist(), c)

#
# The default FCI solver utilize the D2h symmetry to mimic the cylindrical symmetry.
# It may incorrectly remove the initial guess of A2g symmetry with Sz=1 or Sz=-1 .
# direct_spin1_cyl_sym can be used to solve this problem. This module is
# designed for the cylindrical symmetry. By specifying the different Sz and
# wfnsym, the CASSCF can produce three degenerated results.
#
from pyscf.fci import direct_spin1_cyl_sym
nelec = (4, 2)  # Sz=1
mc = mcscf.CASSCF(mf, norb, nelec)
mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
mc.fcisolver.wfnsym = 'A2g'
mc.kernel()

nelec = (3, 3)  # Sz=0
mc = mcscf.CASSCF(mf, norb, nelec)
mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
mc.fcisolver.wfnsym = 'A2g'
mc.kernel()

nelec = (2, 4)  # Sz=-1
mc = mcscf.CASSCF(mf, norb, nelec)
mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
mc.fcisolver.wfnsym = 'A2g'
mc.kernel()
