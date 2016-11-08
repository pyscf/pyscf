#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Use CASSCF orbitals for initial guess of DMRG-SCF
'''

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf

b = 1.2
mol = gto.Mole()
mol.build(
    verbose = 5,
    output = 'out-dmrgscf',
    atom = [['N', (0.,0.,0.)], ['N', (0.,0.,b)]],
    basis = 'cc-pvdz',
    symmetry = True,
)
mf = scf.RHF(mol)
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 6).set(max_cycle_macro=5, max_cycle_micro=1,
                                conv_tol=1e-5, conv_tol_grad=1e-4).run()
#
# Note: the stream operations are applied in the above line.  This one line
# code is equivalent to the following serial statements
#
#mc = mcscf.CASSCF(mf, 6, 6)
#mc.max_cycle_macro = 5
#mc.max_cycle_micro = 1
#mc.conv_tol = 1e-5
#mc.conv_tol_grad = 1e-4
#mc.kernel()
mo = mc.mo_coeff

mol.stdout.write('\n*********** Call DMRGSCF **********\n')
#
# Use CheMPS2 program as the FCI Solver
#
mc = mcscf.CASSCF(mf, 8, 8)
mc.fcisolver = dmrgscf.CheMPS2(mol)
mc.fcisolver.dmrg_e_convergence = 1e-9
emc = mc.mc2step(mo)[0]

#
# Use Block program as the FCI Solver
#
mc = dmrgscf.dmrgci.DMRGSCF(mf, 8, 8)
emc = mc.kernel(mo)[0]
print(emc)
