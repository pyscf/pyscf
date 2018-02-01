#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Given coordinates of grids in real space, evaluate the GTO values and MO
orbital values on these grids.

See also
pyscf/examples/dft/30-ao_value_on_grid.py
pyscf/examples/pbc/30-ao_value_on_grid.py
'''

import numpy
from pyscf import lib, gto, scf

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='sto3g')
mf = scf.RHF(mol).run()

coords = numpy.random.random((100,3))

#
# AO values and MO values on given grids
#
ao = mol.eval_gto('GTOval_sph', coords)
mo = ao.dot(mf.mo_coeff)

#
# AO values and MO gradients on given grids
#
ao_grad = mol.eval_gto('GTOval_ip_sph', coords)  # (3,Ngrids,n) array
mo_grad = [x.dot(mf.mo_coeff) for x in ao_grad]

#
# AO values and gradients and higher order derivatives can be computed
# simultaneously to reduce cost.
#

# deriv=0: orbital value
ao = mol.eval_gto('GTOval_sph_deriv0', coords)

# deriv=1: orbital value + gradients 
ao_p = mol.eval_gto('GTOval_sph_deriv1', coords)  # (4,Ngrids,n) array
ao = ao_p[0]
ao_grad = ao_p[1:4]  # x, y, z

# deriv=2: value + gradients + second order derivatives
ao_p = mol.eval_gto('GTOval_sph_deriv2', coords)  # (10,Ngrids,n) array
ao = ao_p[0]
ao_grad = ao_p[1:4]  # x, y, z
ao_hess = ao_p[4:10] # xx, xy, xz, yy, yz, zz

# deriv=3: value + gradients + second order derivatives + third order
ao_p = mol.eval_gto('GTOval_sph_deriv3', coords)  # (20,Ngrids,n) array
ao = ao_p[0]
ao_grad = ao_p[1:4]   # x, y, z
ao_hess = ao_p[4:10]  # xx, xy, xz, yy, yz, zz
ao_3rd  = ao_p[10:15] # xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz

# deriv=4: ...

