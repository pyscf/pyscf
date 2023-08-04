#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Use callback hooks to control CASSCF solver.
'''

import numpy
from pyscf import gto, scf, mcscf

b = 1.2
mol = gto.Mole()
mol.build(
    verbose = 4,
    atom = [['N', (0.,0.,0.)], ['N', (0.,0.,b)]],
    basis = 'cc-pvdz',
)
mf = scf.RHF(mol)
mf.kernel()

mc = mcscf.CASSCF(mf, 8, 8)

#
# The CASSCF solver provides micro_cycle_scheduler and max_stepsize_scheduler
# hooks for dynamic control of the CASSCF iteration.
# Here we create a micro_cycle_scheduler to call more micro iterations in the
# beginning of the CASSCF iterations.
#
def micro_cycle_scheduler(envs):
    max_micro = envs['casscf'].max_cycle_micro
    if -.01 < envs['de'] and envs['de'] < 0:
        max_micro += int(abs(numpy.log10(-envs['de'])))
    return max_micro
mc.micro_cycle_scheduler = micro_cycle_scheduler

#
# callback function provides another entry to change CASSCF on the fly.
# Eg, here we remove the micro_cycle_scheduler after certain iterations.
#
old_callback = mc.callback
def callback(envs):
    if (envs['imacro'] > 3 and
        'micro_cycle_scheduler' in envs['casscf'].__dict__):
        delattr(envs['casscf'], 'micro_cycle_scheduler')
    if callable(old_callback):
        return old_callback(envs)
mc.callback = callback

mc.kernel()

