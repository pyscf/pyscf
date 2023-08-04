#!/usr/bin/env python

import numpy
from pyscf import gto
from pyscf.dft import numint

'''
Evaluate AO functions on given grid coordinates.

See also
pyscf/examples/gto/24-ao_value_on_grid.py
pyscf/examples/pbc/30-ao_value_on_grid.py
'''

mol = gto.M(
    verbose = 0,
    atom = '''
    o    0    0.       0.
    h    0    -0.757   0.587
    h    0    0.757    0.587''',
    basis = '6-31g')

# Uniform grids
coords = []
for ix in numpy.arange(-10, 10, 1.):
    for iy in numpy.arange(-10, 10, 1.):
        for iz in numpy.arange(-10, 10, 1.):
            coords.append((ix,iy,iz))
coords = numpy.array(coords)

# AO value
ao_value = numint.eval_ao(mol, coords)
print('13 basis, 8000 xyz  %s' % str(ao_value.shape))

# AO value and its gradients
ao_value = numint.eval_ao(mol, coords, deriv=1)
print('13 basis, 8000 xyz  %s' % str(ao_value.shape))

