#!/usr/bin/env python

import numpy
from pyscf import gto, dft
from pyscf.dft import numint

'''
Evaluate exchange-correlation functional and its potential on given grid
coordinates.
'''

mol = gto.M(
    verbose = 0,
    atom = '''
    o    0    0.       0.
    h    0    -0.757   0.587
    h    0    0.757    0.587''',
    basis = '6-31g')

mf = dft.RKS(mol)
mf.kernel()
dm = mf.make_rdm1()

# Use default mesh grids and weights
coords = mf.grids.coords
weights = mf.grids.weights
ao_value = numint.eval_ao(mol, coords, deriv=1)
# The first row of rho is electron density, the rest three rows are electron
# density gradients which are needed for GGA functional
rho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
print(rho.shape)

#
# Evaluate XC functional one by one.
# Note: to evaluate only correlation functional, put ',' before the functional name
#
ex, vx = dft.libxc.eval_xc('B88', rho)[:2]
ec, vc = dft.libxc.eval_xc(',P86', rho)[:2]
print('Exc = %.12f' % numpy.einsum('i,i,i->', ex+ec, rho[0], weights))

#
# Evaluate XC functional together
#
exc, vxc = dft.libxc.eval_xc('B88,P86', rho)[:2]
print('Exc = %.12f' % numpy.einsum('i,i,i->', exc, rho[0], weights))

#
# Evaluate XC functional for user specified functional
#
exc, vxc = dft.libxc.eval_xc('.2*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN', rho)[:2]
print('Exc = %.12f  ref = -7.520014202688' % numpy.einsum('i,i,i->', exc, rho[0], weights))

