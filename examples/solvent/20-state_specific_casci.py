#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Using solvent model for excited state CASCI calculations.
'''

from pyscf import gto, scf, mcscf
from pyscf import solvent

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            basis='ccpvdz',
            verbose=4)

# Initialize the reference HF calculation
mf = solvent.ddCOSMO(scf.RHF(mol)).run()

# Compute 5 states in CASCI. Slow solvent was assumed. Its effects are frozen
# at HF level.
mc = solvent.ddCOSMO(mcscf.CASCI(mf, 4, 4), dm=mf.make_rdm1())
mc.fcisolver.nstates = 5
mc.run()
es_1 = mc.e_tot

# Compute 5 states in CASCI, while the solvent only responses to the first
# excited state.
mc = solvent.ddCOSMO(mcscf.CASCI(mf, 4, 4))
mc.fcisolver.nstates = 5
mc.with_solvent.state_id = 1
mc.run()
es_2 = mc.e_tot

# Compute 5 states in CASCI, while the solvent only responses to the third
# excited state.
mc = solvent.ddCOSMO(mcscf.CASCI(mf, 4, 4))
mc.fcisolver.nstates = 5
mc.with_solvent.state_id = 3
mc.run()
es_3 = mc.e_tot

print('           Slow solvent    Fast solvent for state 1   Fast solvent for state 3')
for i in range(len(es_1)):
    print('State = %d  %.12g  %.12g             %.12g' % (i, es_1[i], es_2[i], es_3[i]))
