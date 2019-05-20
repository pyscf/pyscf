#!/usr/bin/env python

'''
Optimize molecular geometry within the environment of QM/MM charges.
'''

import numpy
from pyscf import gto, scf, cc, qmmm

mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
            ''',
            basis='3-21g')

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * -.001
mf = qmmm.mm_charge(scf.RHF(mol), coords, charges)
#mf.verbose=4
#mf.kernel()
mol1 = mf.Gradients().optimizer(solver='berny').kernel()
# or
#from pyscf.geomopt import berny_solver
#mol1 = berny_solver.optimize(mf)

mycc = cc.CCSD(mf)
mol1 = mycc.Gradients().optimizer().kernel()
# or
#from pyscf.geomopt import geometric_solver
#mol1 = geometric_solver.optimize(mycc)
