#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
geometry optimization for TDDFT with solvent
'''

import sys
from pyscf import gto
from pyscf import __all__

verify_windows = '--pyscf-verify-windows' in sys.argv

mol = gto.M(
    atom = '''
H     0.   0.   .917
F     0.   0.   0.
''',
basis = '631g')

#
# It's preferable to use equilibrium solvation in geometry optimization
#
mf = mol.RHF().ddCOSMO().run()
td = mf.TDA().ddCOSMO()
td.with_solvent.equilibrium_solvation = True

try:
    mol_eq = td.nuc_grad_method().as_scanner(state=2).optimizer().kernel()
except ModuleNotFoundError:
    if verify_windows:
        # TDDFT geometry optimization relies on the optional geometric package.
        print('Skipping TDDFT solvent geomopt example during Windows verification because geometric is not installed.')
        raise SystemExit(0)
    raise

