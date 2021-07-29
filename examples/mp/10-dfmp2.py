#!/usr/bin/env python
#

'''
Example calculation with the native DF-MP2 code.
'''

from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.mp.dfmp2_native import DFMP2

mol = Mole()
mol.atom = '''
H   0.0   0.0   0.0
F   0.0   0.0   1.1
'''
mol.basis = 'cc-pVDZ'
mol.spin = 0
mol.build()

mf = RHF(mol).run()
DFMP2(mf).run()