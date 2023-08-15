#!/usr/bin/env python
#
# Author: Nikolay A. Bogdanov <n.bogdanov@inbox.lv>
#

from pyscf import gto
from pyscf import scf
from pyscf.symm import msym

'''
Use external libmsym library to run calculations with non-abelian symmetry
'''


mol = gto.Mole()
mol.build(
    verbose=4,
    atom='''
 C  0.000000  0.000000  0.000000
 H  0.000000 -0.000003  1.089000
 H  1.026719  0.000003 -0.363000
 H -0.513358 -0.889165 -0.363002
 H -0.513361  0.889165 -0.362998
    ''',
    basis='cc-pvdz',
    spin=0,
    symmetry=True,
)

mf = scf.RHF(mol)
e = mf.kernel()
print('E = %.15g' % e)


mol_msym = msym.gen_mol_msym(mol, tol=1e-6)
mf_msym = scf.RHF(mol_msym)
e = mf_msym.kernel()
print('E = %.15g' % e)

dm_msym = mf_msym.make_rdm1()

e = mf.kernel(dm_msym)
print('E = %.15g' % e)
