#!/usr/bin/env python

'''
Optimize molecular geometry with or w/o ghost atoms.
(In testing)
'''

import sys
verify_windows = '--pyscf-verify-windows' in sys.argv
try:
    from pyscf.geomopt import berny_solver
except ModuleNotFoundError:
    if verify_windows:
        # Ghost-atom optimization relies on the optional berny solver.
        print('Skipping ghost-atom geomopt example during Windows verification because berny is not installed.')
        raise SystemExit(0)
    raise
from pyscf import gto, scf

mol = gto.M(atom='''
GHOST-O  0.   0.       0.
H  0.   -0.757   0.587
H  0.   0.757    0.587 ''',
            basis='631g')
mf = scf.RHF(mol)
berny_solver.optimize(mf, include_ghost=True)

berny_solver.optimize(mf, include_ghost=False)
