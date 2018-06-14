#!/usr/bin/env python

'''
Optimize molecular geometry with or w/o ghost atoms.
(In testing)
'''

from pyscf.geomopt import berny_solver
from pyscf import gto, scf

mol = gto.M(atom='''
GHOST-O  0.   0.       0.
H  0.   -0.757   0.587
H  0.   0.757    0.587 ''',
            basis='631g')
mf = scf.RHF(mol)
berny_solver.optimize(mf, include_ghost=True)

berny_solver.optimize(mf, include_ghost=False)
