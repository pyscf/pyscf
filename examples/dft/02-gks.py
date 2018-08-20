#!/usr/bin/env python

'''
GKS for general spin-orbital Kohn-Sham method.

See also pyscf/examples/scf/02-ghf.py
'''

from pyscf import gto, dft

#
# 1. real GKS
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

mf = dft.GKS(mol)
mf.xc = 'camb3lyp'
mf.kernel()

#
# 2. complex GKS
#
mf = dft.GKS(mol)
mf.xc = 'camb3lyp'
dm = mf.get_init_guess() + 0j
dm[0,:] += .1j
dm[:,0] -= .1j
mf.kernel(dm0=dm)

