#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
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

tmpchk = tempfile.NamedTemporaryFile()

mol = gto.Mole()
mol.atom = 'C 0 0 0; C 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 6)
mc.chkfile = tmpchk.name
mc.max_cycle_macro = 1
mc.kernel()

#######################################################################
#
# Assuming the CASSCF was interrupted.  Intermediate data were saved in
# tmpchk file.  Here we read the chkfile to restart the previous calculation.
#
#######################################################################
mol = gto.Mole()
mol.atom = 'C 0 0 0; C 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()

mc = mcscf.CASSCF(scf.RHF(mol), 6, 6)
mo = lib.chkfile.load(tmpchk.name, 'mcscf/mo_coeff')
mc.kernel(mo)
