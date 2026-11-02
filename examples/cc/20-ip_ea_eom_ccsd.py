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

# The reference Python RCCSD implementation also exposes perturbative
# EOM-CCSD* corrections from matched left and right eigenvectors.
from pyscf.cc import rccsd_slow
slowcc = rccsd_slow.RCCSD(mf).run()
eip, rip = slowcc.ipccsd(nroots=1)
_, lip = slowcc.ipccsd(nroots=1, left=True)
eea, rea = slowcc.eaccsd(nroots=1)
_, lea = slowcc.eaccsd(nroots=1, left=True)
print('IP/EA-EOM-CCSD*', slowcc.ipccsd_star_contract(eip, rip, lip),
      slowcc.eaccsd_star_contract(eea, rea, lea))


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
