#!/usr/bin/env python

'''
Optimize the geometry of the excited states

Note when optiming the excited states, states may flip and this may cause
convergence issue in geometry optimizer.
'''

from pyscf import gto
from pyscf import scf
from pyscf import ci, tdscf, mcscf
from pyscf import geomopt


mol = gto.Mole()
mol.atom="N; N 1, 1.1"
mol.basis= "6-31g"
mol.build()
mol1 = mol.copy()

mf = scf.RHF(mol).run()

mc = mcscf.CASCI(mf, 4,4)
mc.fcisolver.nstates = 3
excited_grad = mc.nuc_grad_method().as_scanner(state=2)
mol1 = excited_grad.optimizer().kernel()
# or
#mol1 = geomopt.optimize(excited_grad)


td = tdscf.TDHF(mf)
td.nstates = 5
excited_grad = td.nuc_grad_method().as_scanner(state=4)
mol1 = excited_grad.optimizer().kernel()
# or
#mol1 = geomopt.optimize(excited_grad)


myci = ci.CISD(mf)
myci.nstates = 2
excited_grad = myci.nuc_grad_method().as_scanner(state=1)
mol1 = excited_grad.optimizer().kernel()
# or
#geomopt.optimize(excited_grad)

