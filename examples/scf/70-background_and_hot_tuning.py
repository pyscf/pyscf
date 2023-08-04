#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Put SCF calculation in the background.  Tune the SCF solver at runtime.
'''

import time
from pyscf import gto, scf, lib

mol = gto.Mole()
mol.build(
    verbose = 4,
    atom = [('H', (i*2.4, 0, 0)) for i in range(50)],
    basis = 'sto6g',
)
mf = scf.RHF(mol)
#
# Switch off diis and incore JK contraction to slow down the SCF iteration.
#
mf.max_memory = 10
mf.diis = None

#
# Use function bg (background thread) to do non-blocking function call.
#
b = lib.bg(mf.kernel)

#
# Modify the mf object during SCF iteration.
# Wait 2 seconds to ensure that the modification happens during the SCF iteration.
#
time.sleep(2)
print('\n ** Set max_cycle to 10\n')
mf.max_cycle = 10

#
# Load the intermediate SCF result to mf object from chkfile.  One can use the
# intermediate data before the SCF finishing
#
time.sleep(1)
mf.update_from_chk()
from pyscf.tools import molden
molden.from_scf(mf, 'unconverged_scf.molden')
print('\n ** Write SCF intemediates to molden file\n')
e_scf = b.get()
print('E(HF) = %s ' % e_scf)

