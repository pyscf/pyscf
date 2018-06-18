#!/usr/bin/env python

# Author: Qiming Sun <osirpt.sun@gmail.com>

'''
Control level shift during the SCF iterations.

The level shift schedule can be applied on the get_fock method.  This example
shows two kinds of schedules to decrease the level shift during SCF iterations.
See also the implementation dynamic_level_shift_ in pyscf/scf/addons.py
'''

import os
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import dft


mol = gto.Mole()
mol.verbose = 4
mol.atom = [
 ["C",  ( 0.000000,  0.418626, 0.000000)],
 ["H",  (-0.460595,  1.426053, 0.000000)],
 ["O",  ( 1.196516,  0.242075, 0.000000)],
 ["N",  (-0.936579, -0.568753, 0.000000)],
 ["H",  (-0.634414, -1.530889, 0.000000)],
 ["H",  (-1.921071, -0.362247, 0.000000)]
]
mol.basis = '631g**'
mol.build()

#
# scf.addons.dynamic_level_shift_ implement a schedule to reduce the level shift
# according to the energy changes during the SCF iteration.  The level shift
# will be removed when the energy change approaches 0.
#
mf = dft.RKS(mol)
scf.addons.dynamic_level_shift_(mf, factor=0.5)
mf.kernel()

#
# Schedule 2: Reduce the level shift by half in the first 5 SCF iteration then
# completely remove the level shift in the rest iterations.
#
old_get_fock = mf.get_fock
level_shift0 = 0.5
def get_fock(h1e, s1e, vhf, dm, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if cycle < 5:
        level_shift_factor = level_shift0 * .5 ** cycle
        print('Set level shift to %g' % level_shift_factor)
    else:
        level_shift_factor = 0
    return old_get_fock(h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle,
                        level_shift_factor, damp_factor)
mf.get_fock = get_fock
mf.kernel()
