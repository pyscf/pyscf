#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
from pyscf import gto, scf, mcscf
from pyscf import lib

'''
Restart CASSCF from previous calculation.

There is no "restart" keyword for CASSCF solver.  The CASSCF solver is
completely controlled by initial guess.  So we can mimic the restart feature
by providing proper initial guess from previous calculation.

We need assign the .chkfile a string to indicate the file where to save the
CASSCF intermediate results.  Then we can "restart" the calculation from the
intermediate results.
'''

mol = gto.Mole()
mol.atom = 'C 0 0 0; C 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()

chkname = os.path.join(lib.param.TMPDIR, '13-restart.chk')
mf = scf.RHF(mol)
mf.chkfile = chkname
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 6)
mc.chkfile = chkname
mc.max_cycle_macro = 1
mc.kernel()

#######################################################################
#
# Assuming the CASSCF was interrupted.  Intermediate data were saved in
# chkname file.  Here we read the chkfile to restart the previous calculation.
#
#######################################################################
mol = gto.Mole()
mol.atom = 'C 0 0 0; C 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()

mc = mcscf.CASSCF(scf.RHF(mol), 6, 6)
mo = lib.chkfile.load(chkname, 'mcscf/mo_coeff')
mc.kernel(mo)

# Assuming you lose all memory about the previous calculation.
# Restart the calculation with chkfile only.
mol, mcdata = mcscf.chkfile.load_mcscf(chkname)
mc = mcscf.CASSCF(mol, mcdata['ncas'], mcdata['nelecas']).update_from_chk(chkname)
mc.kernel()
