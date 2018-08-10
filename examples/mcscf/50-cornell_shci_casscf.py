#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Use Cornell SHCI program as the active space solver for CASSCF
'''

from pyscf import gto, scf, mcscf, cornell_shci

b = 1.2
mol = gto.M(
    atom = 'N 0 0 0; N 0 0 %f'%b,
    basis = 'cc-pvdz',
    verbose = 4
)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 8, 6).run()

#
# Use SHCI as the FCI Solver for CASSCF. It's recommended to initialize
# SHCI-CASSCF using function cornell_shci.SHCISCF. It includes special
# initialization.
#
mc = cornell_shci.SHCISCF(mf, 8, 6)
mc.kernel()

#
# Use SHCI as the FCI Solver for CASCI.
#
mc = mcscf.CASCI(mf, 8, 6)
mc.fcisolver = cornell_shci.SHCI(mol)
mc.kernel()
