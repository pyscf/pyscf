#!/usr/bin/env python

'''
:func:`scf.addons.remove_linear_dep_` discards the small eigenvalues of overlap
matrix.  This reduces the number of MOs from 50 to 49.  The problem size of
the following CCSD method is 49.
'''

from pyscf import gto, scf, cc
mol = gto.Mole()
mol.atom = [('H', 0, 0, .5*i) for i in range(20)]
mol.basis = 'ccpvdz'
mol.verbose = 4
mol.build()
# Without handling the linear dependency in basis, HF and CCSD can produce
# incorrect results
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()

mf = scf.addons.remove_linear_dep_(mol.RHF()).run()
mycc = cc.CCSD(mf).run()

