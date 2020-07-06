#!/usr/bin/env python

'''
Compute electron density in real space.

See also the example dft/31-xc_value_on_grid.py
'''

import numpy as np
import pyscf


mol = pyscf.M(
    atom = '''
    O    0    0.       0.
    H    0    -0.757   0.587
    H    0    0.757    0.587''',
    basis = 'ccpvdz')

cc = mol.CCSD().run()

# Put 100x100x100 uniform grids in a box. The unit for coordinates is Bohr
xs = np.arange(-5, 5, .1)
ys = np.arange(-5, 5, .1)
zs = np.arange(-5, 5, .1)
grids = pyscf.lib.cartesian_prod([xs, ys, zs])

# Compute density matrix and evaluate the electron density with it. 
# Note the density matrix has to be in AO basis
dm = cc.make_rdm1(ao_repr=True)
ao_value = pyscf.dft.numint.eval_ao(mol, grids, deriv=0)
rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm)

print('The shape of the electron density', rho.shape)
