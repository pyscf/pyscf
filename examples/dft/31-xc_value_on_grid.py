#!/usr/bin/env python

import numpy
from pyscf import gto, dft
from pyscf.dft import numint

'''
Evaluate exchange-correlation functional and itr potential on given grid
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
ao_value = numint.eval_ao(mol, coords, isgga=True)
# The first row of rho is electron density, the rest three rows are electron
# density gradients which are needed for GGA functional
rho = numint.eval_rho(mol, ao_value, dm, isgga=True)
print(rho.shape)
sigma = numpy.einsum('ip,ip->p', rho[1:], rho[1:])

# See pyscf/dft/vxc.py for the XC functional ID
x_id = dft.XC_GGA_X_B88
c_id = dft.XC_GGA_C_P86
ex, vx, vx_sigma = numint.eval_x(x_id, rho[0], sigma)
ec, vc, vc_sigma = numint.eval_c(c_id, rho[0], sigma)
print('Exc = %.12f' % numpy.einsum('i,i,i->', ex+ec, rho[0], weights))
print('Vxc on each grid %s' % str(vx.shape))
print('Vxc_sigma on each grid %s' % str(vx.shape))

