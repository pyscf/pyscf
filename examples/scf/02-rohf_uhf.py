#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
scf.RHF function can pick restricted closed shell HF or restricted open shell
HF solver, depending on the system spin.  scf.ROHF only creates restricted
open shell solver.
'''

from pyscf import gto, scf

#
# 1. open-shell system
#
mol = gto.M(
    atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
    basis = 'ccpvdz',
    charge = 1,
    spin = 1  # = 2S = spin_up - spin_down
)

#
# == ROHF solver
#
mf = scf.RHF(mol)
mf.kernel()

mf = scf.ROHF(mol)
mf.kernel()

mf = scf.UHF(mol)
mf.kernel()


#
# 2. closed-shell system
#
mol = gto.M(
    atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587''',
    basis = 'ccpvdz',
)

#
# Using restricted closed shell solver
#
mf = scf.RHF(mol)
mf.kernel()

#
# Using restricted open shell solver
#
mf = scf.ROHF(mol)
mf.kernel()

mf = scf.UHF(mol)
mf.kernel()
