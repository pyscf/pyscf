#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD frozen core
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()

#
# Freeze the inner most two orbitals.
#
mycc = cc.CCSD(mf)
mycc.frozen = 2
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

#
# Auto-generate the number of core orbitals to be frozen.
# In this case, it would be 1.
#
mycc = cc.CCSD(mf)
from pyscf.data import elements
mycc.frozen = elements.chemcore(mol)
print('Number of core orbital frozen: %d' % mycc.frozen)
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)
# Shorter code, it's identical to the above one
mycc.set_frozen()
print('Number of core orbital frozen: %d' % mycc.frozen)
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)
# Note: for GCCSD, the frozen is 2 times that of RCCSD/UCCSD
mycc = cc.GCCSD(mf)
mycc.set_frozen()
print('Number of core orbital frozen: %d' % mycc.frozen)
mycc.kernel()
print('GCCSD correlation energy', mycc.e_corr)


#
# Freeze orbitals based on the list of indices.
#
mycc = cc.CCSD(mf)
mycc.frozen = [0,1,16,17,18]
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

#
# Freeze orbitals based on energy window (in a.u.).
#
mycc.set_frozen(method='window', window=(-1000.0, 4.1))
#print(mycc._scf.mo_energy)
print('List of orbital frozen: ', mycc.frozen)
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

#
# set_frozen will reduce the number of frozen orbitals 
# when ECP exists, and return 0 if 
# number of elec screened by ECP > number of chemical core electrons
#
mol = gto.M(
    atom = 'Xe 0 0 0',
    basis = 'cc-pvtz-dk')
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
mycc.set_frozen()
print('Number of core orbital frozen: %d' % mycc.frozen)
mol.set(basis='def2-svp', ecp='def2-svp').build()
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
mycc.set_frozen()
print('Number of core orbital frozen: %d' % mycc.frozen)

