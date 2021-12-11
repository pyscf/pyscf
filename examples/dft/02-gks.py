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
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
    basis = 'ccpvdz',
    charge = 1,
    spin = 1,  # = 2S = spin_up - spin_down
    verbose = 4
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

