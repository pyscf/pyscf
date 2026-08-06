#!/usr/bin/env python

'''
Edmiston-Ruedenberg (ER) localization and stability analysis

This example demonstrates ER localization using both the CIAH and BFGS algorithms.
It also illustrates stability analysis of the localized orbitals.
'''

import numpy
from pyscf import gto, scf
from pyscf.lib import logger
from pyscf.lo import ER
from pyscf.lo.tools import findiff_grad, findiff_hess

mol = gto.Mole()
mol.atom = '''
     O   0.    0.     0.2
     H    0.   -0.5   -0.4
     H    0.    0.7   -0.2
  '''
mol.basis = 'ccpvdz'
mol.build()
mf = scf.RHF(mol).run()

log = logger.new_logger(mol, verbose=6)

mo = mf.mo_coeff[:,:mol.nelectron//2]
mlo = ER(mol, mo)

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

# localization + stability check using BFGS
mlo.algorithm = 'bfgs'
mlo.kernel()

while True:
    mo, stable = mlo.stability(return_status=True)
    if stable:
        break
    mlo.kernel(mo)

