#!/usr/bin/env python

'''
Optimize molecular geometry within the environment of QM/MM charges.
'''

from pyscf import gto, scf
from pyscf.geomopt import berny_solver
from pyscf.geomopt import geometric_solver

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            basis='3-21g')

mf = scf.RHF(mol)

# Run analyze function in callback
def cb(envs):
    mf = envs['g_scanner'].base
    mf.analyze(verbose=4)

#
# Method 1: Pass callback to optimize function
#
geometric_solver.optimize(mf, callback=cb)

berny_solver.optimize(mf, callback=cb)

#
# Method 2: Add callback to geometry optimizer
#
opt = mf.nuc_grad_method().as_scanner().optimizer()
opt.callback = cb
opt.kernel()
