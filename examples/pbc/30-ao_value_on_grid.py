#!/usr/bin/env python

'''
Given coordinates of grids in real space, evaluate the pbc-GTO values
on these grids.

See also
pyscf/examples/gto/24-ao_value_on_grid.py
pyscf/examples/pbc/30-ao_value_on_grid.py
'''

import numpy
from pyscf.pbc import gto, scf, dft

cell = gto.Cell()
cell.atom = '''
He 0 0 1
He 1 0 1
'''
cell.basis = 'ccpvdz'
cell.a = numpy.eye(3) * 4
cell.verbose = 4
cell.build()

coords = numpy.random.random((100,3))
nks = [2,2,2]
kpts = cell.make_kpts(nks)

#
# AO value for a single k-point
#
ao = dft.numint.eval_ao(cell, coords, kpt=kpts[1], deriv=0)
# or
ao = cell.pbc_eval_gto('GTOval', coords, kpt=kpts[1])
# Note cell.eval_gto and cell.pbc_eval_gto are different.  Their relations are
# like cell.intor and cell.pbc_intor.  cell.eval_gto only returns the value of
# non-PBC GTOs on the given grids while cell.pbc_eval_gto is the PBC-GTO
# version which includes the lattice summation over repeated images

#
# AO value for k-points, the first index runs over k-points
#
ao_kpts = dft.numint.eval_ao_kpts(cell, coords, kpts=kpts, deriv=0)
print(numpy.linalg.norm(ao_kpts[1] - ao) < 1e-12)

#
# AO value and gradients for a single k-point
#
ao_p = dft.numint.eval_ao(cell, coords, kpt=kpts[1], deriv=1)
ao = ao_p[0]
ao_grad = ao_p[1:4]
# Inclusion of high order derivatives
ao_p = dft.numint.eval_ao(cell, coords, kpt=kpts[1], deriv=2)
ao_p = dft.numint.eval_ao(cell, coords, kpt=kpts[1], deriv=3)
ao_p = dft.numint.eval_ao(cell, coords, kpt=kpts[1], deriv=4)

#
# AO value and gradients for k-points
#
ao_p = dft.numint.eval_ao_kpts(cell, coords, kpts=kpts, deriv=1)
ao_kpts = [ao[0] for ao in ao_p]
ao_grad_kpts = [ao[1:4] for ao in ao_p]

#
# AO value and gradients and hessian for k-points
#
ao_p = dft.numint.eval_ao_kpts(cell, coords, kpts=kpts, deriv=2)
ao_kpts = [ao[0] for ao in ao_p]         # x, y, z
ao_grad_kpts = [ao[1:4] for ao in ao_p]  # xx, xy, xz, yy, yz, zz
ao_hess_kpts = [ao[4:10] for ao in ao_p] # xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
# Inclusion of high order derivatives
ao_p = dft.numint.eval_ao_kpts(cell, coords, kpts=kpts, deriv=3)
ao_p = dft.numint.eval_ao_kpts(cell, coords, kpts=kpts, deriv=4)
