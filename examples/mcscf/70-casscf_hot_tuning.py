#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Tune CASSCF solver on the fly.

However, it is unrecommended to tune CASSCF solver on the runtime unless you
know exactly what you're doing.
'''

from pyscf import gto, scf, mcscf

b = 1.2
mol = gto.Mole()
mol.build(
    verbose = 5,
    atom = [['N', (0.,0.,0.)], ['N', (0.,0.,b)]],
    basis = 'cc-pvdz',
)
mf = scf.RHF(mol)
mf.kernel()

#
# This step creates a hook on CASSCF callback function.  It allows the CASSCF
# solver reading the contents of config and update the solver itself in every
# micro iteration.
#
# Then start the casscf solver.
#
mc = mcscf.hot_tuning_(mcscf.CASSCF(mf, 6, 6), 'config')
mc.kernel()

#
# The solver finishes quickly for this system since it is small.  Assuming the
# system is large and the optimization iteration processes slowly,  we can
# modify the content of config during the optimization, to change the behavior
# of CASSCF solver.  Eg
#
# 1. We can set the "frozen" attribute in the config file, to avoid the
# orbital rotation over optimized.  After a few optimization cycles, we can
# reset "frozen" to null, to restore the regular CASSCF optimization.
#
# 2. We can tune ah_start_cycle to increase the AH solver accuracy.  Big value
# in ah_start_cycle can postpone the orbital rotation with better AH solution.
# It may help predict a better step in orbital rotation function.
#
