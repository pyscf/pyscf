#!/usr/bin/env python

'''
Ground-state, EOM-EE-GCCSD and IP/EA-EOM-CCSD for singlet (RHF) and triplet (UHF) O2.
'''

from pyscf import gto, scf, cc

# Singlet

mol = gto.Mole()
mol.verbose = 5
mol.unit = 'A'
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
eee,cee = mycc.eeccsd(nroots=1)

# S->S excitation
eS = mycc.eomee_ccsd_singlet(nroots=1)[0]
# S->T excitation
eT = mycc.eomee_ccsd_triplet(nroots=1)[0]


# Triplet

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

# EOM-GCCSD
mf = mf.to_ghf()
mycc = mf.CCSD()
ecc, t1, t2 = mycc.kernel()
e,v = mycc.ipccsd(nroots=6)
e,v = mycc.eaccsd(nroots=6)
e,v = mycc.eeccsd(nroots=6)
