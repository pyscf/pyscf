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

geometric_solver.optimize(mf, callback=cb)

berny_solver.optimize(mf, callback=cb)

