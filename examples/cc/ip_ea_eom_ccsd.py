#!/usr/bin/env python

'''
Ground-state and IP/EA-EOM-CCSD for singlet (RHF) and triplet (UHF) O2.
'''

from pyscf import gto, scf, cc

'''Singlet'''

mol = gto.Mole()
mol.verbose = 5
mol.atom = 'A'
mol.atom = 'O 0 0 0; O 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()

mf = scf.RHF(mol)
mf.verbose = 7
mf.scf()

mycc = cc.RCCSD(mf)
mycc.verbose = 7
mycc.ccsd()

eip,cip = mycc.ipccsd(nroots=1)
eea,cea = mycc.eaccsd(nroots=1)

'''Triplet'''

mol.spin = 2
mol.build()

mf = scf.UHF(mol)
mf.verbose = 7
mf.scf()

mycc = cc.UCCSD(mf)
mycc.verbose = 7
mycc.ccsd()

eip,cip = mycc.ipccsd(nroots=1)
eea,cea = mycc.eaccsd(nroots=1)

