#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Slow solvent and fast slovent in TDDFT calculations.
'''

from pyscf import gto
from pyscf import __all__

mol = gto.M(
    atom = '''
H     0.   0.   .917
F     0.   0.   0.
''',
basis = '631g')

#
# Solvent does not respond to the change of electronic structure in vertical
# excitation. The calculation can be started with an SCF with fully relaxed
# solvent and followed by a regular TDDFT method
#
mf = mol.RHF().ddCOSMO().run()
td = mf.TDA()
td.kernel()


#
# Equilibrium solvation allows the solvent rapidly responds to the electronic
# structure of excited states. The system should converge to equilibrium
# between solvent and the excited state of the solute.
#
mf = mol.RHF().ddCOSMO().run()
td = mf.TDA().ddCOSMO()
td.with_solvent.equilibrium_solvation = True
td.kernel()

#
# Switch off the fast solvent
#
td.with_solvent.equilibrium_solvation = False
td.kernel()
