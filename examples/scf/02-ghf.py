#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
scf.GHF, real, complex.
'''

from pyscf import gto, scf

#
# 1. real GHF
#
mol = gto.M(
    atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587''',
    basis = 'ccpvdz',
    charge = 1,
    spin = 1  # = 2S = spin_up - spin_down
)

mf = scf.GHF(mol)
mf.kernel()

#
# 2. complex GHF
#
mf = scf.GHF(mol)
dm = mf.get_init_guess() + 0j
dm[0,:] += .05j
dm[:,0] -= .05j
mf.kernel(dm0=dm)
