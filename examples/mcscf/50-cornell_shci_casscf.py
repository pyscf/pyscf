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
    verbose = 4,
    symmetry = 1
)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 8, 10).run()

#
# Use SHCI as the FCI Solver for CASSCF. It's recommended to create the
# SHCI-CASSCF object using function cornell_shci.SHCISCF. It initializes
# optimized parameters for SHCI-CASSCF method.
#
mc = cornell_shci.SHCISCF(mf, 8, 10).run()

#
# Use SHCI as the FCI Solver for CASCI.
#
mc = mcscf.CASCI(mf, 8, 10)
mc.fcisolver = cornell_shci.SHCI(mol)
mc.kernel()

#
# SHCI input parameters are stored in the attribute config of SHCI class.
# See the shci project (https://github.com/jl2922/shci) for more details of
# the input configs.
#
mc.fcisolver.config['eps_vars'] = [1e-6]
mc.fcisolver.config['eps_pt'] = 1e-10
mc.kernel()
