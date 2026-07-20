#!/usr/bin/env python

'''
PM localization for PBC k-point orbitals

This example demonstrates PM localization using both the CIAH and BFGS algorithms.
It also illustrates stability analysis of the localized orbitals.

Reference:
    Fast Generation of Pipek–Mezey Wannier Functions via the Co-Iterative Augmented Hessian Method,
    G. Yang and H. Ye, DOI: 10.1021/acs.jctc.6c00280.
'''

import numpy
from pyscf.lib import logger
from pyscf.pbc import gto, scf
from pyscf.pbc.lo import KPM
from pyscf.lo.tools import findiff_grad, findiff_hess

cell = gto.Cell()
cell.atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
'''
cell.a = numpy.eye(3) * 5
cell.basis = 'ccpvdz'
cell.build()

kpts = cell.make_kpts([3,1,1])

mf = scf.KRHF(cell, kpts=kpts).density_fit().run()

log = logger.new_logger(cell, verbose=6)

mo = [mok[:,:cell.nelectron//2] for mok in mf.mo_coeff]
mlo = KPM(cell, mo, kpts)

# Validate gradient and Hessian against finite difference
g, h_op, hdiag = mlo.gen_g_hop()

h = numpy.zeros((mlo.pdim,mlo.pdim))
x0 = mlo.zero_uniq_var()
for i in range(mlo.pdim):
    x0[i] = 1
    h[:,i] = h_op(x0)
    x0[i] = 0

def func(x):
    u = mlo.extract_rotation(x)
    f = mlo.cost_function(u)
    if mlo.maximize:
        return -f
    else:
        return f

def fgrad(x):
    u = mlo.extract_rotation(x)
    return mlo.get_grad(u)

g_num = findiff_grad(func, x0)
h_num = findiff_hess(fgrad, x0)
hdiag_num = numpy.diag(h_num)

log.info('Grad  error: %.3e', abs(g-g_num).max())
log.info('Hess  error: %.3e', abs(h-h_num).max())
log.info('Hdiag error: %.3e', abs(hdiag-hdiag_num).max())

# localization + stability check using CIAH
mlo.verbose = 4
mlo.algorithm = 'ciah'
mlo.kernel()

while True:
    mo, stable = mlo.stability(return_status=True)
    if stable:
        break
    mlo.kernel(mo)

# localization + Jacobi-based stability check using BFGS
mlo.algorithm = 'bfgs'
mlo.kernel()

while True:
    mo, stable = mlo.stability_jacobi(return_status=True)
    if stable:
        break
    mlo.kernel(mo)
