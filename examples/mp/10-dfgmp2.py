#!/usr/bin/env python

'''
Example calculation using the DF-GMP2 code.

Relevant examples:
10-dfmp2.py 10-dfump2.py 10-dfgmp2.py 11-dfmp2-density.py 12-dfump2-natorbs.py
'''

from pyscf import gto, scf, mp
from pyscf.mp import GMP2
from pyscf.mp.dfgmp2 import DFGMP2

mol = gto.Mole()
mol.atom = [
    ['Li', (0., 0., 0.)],
    ['H', (1., 0., 0.)]]
mol.basis = 'cc-pvdz'
mol.build()

mf = mol.GHF().run()
mymp = mf.DFGMP2()
mymp.kernel()

# When mean-field calculation is a density fitting HF method, the .DFMP2()
# method is identical to the standard .MP2 method
mf = mf.density_fit().run()
mf.MP2().run()

# DF-GMP2 supports complex orbitals
dm = mf.get_init_guess() + 0j
dm[0,:] += .1j
dm[:,0] -= .1j
mf.kernel(dm0=dm)
mymp = DFGMP2(mf)
mymp.kernel()
