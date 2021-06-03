#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Geometry optimization with solvent model
'''

from pyscf import gto, scf, dft
from pyscf import solvent
from pyscf.geomopt import geometric_solver

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)
mf = solvent.ddCOSMO(scf.RHF(mol))
new_mol = geometric_solver.optimize(mf)

