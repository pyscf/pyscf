#!/usr/bin/env python

'''
Use "scanner" function to compute the molecule dissociation curve and the
force on the curve.  Note the force is based on the atomic unit.
'''

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf

bond = np.arange(0.8, 5.0, .1)
energy = []
force = []
mol = gto.Mole(atom=[['N', 0, 0, -0.4],
                     ['N', 0, 0,  0.4]],
               basis='ccpvdz')

mf_grad_scan = scf.RHF(mol).nuc_grad_method().as_scanner()
for r in reversed(bond):
    e_tot, grad = mf_grad_scan([['N', 0, 0, -r / 2],
                                ['N', 0, 0,  r / 2]])
    energy.append(e_tot)
    force.append(grad[0,2])

plt.plot(bond, energy[::-1])
plt.show()

plt.plot(bond, force[::-1])
plt.show()


# When a molecular geometry is input to the scanner, it uses the SAME unit as
# the one in the previous calculation.
mol = gto.Mole(atom='N; N 1 1.2',
               unit='Ang',
               basis='ccpvdz')
mf_grad_scan = scf.RHF(mol).nuc_grad_method().as_scanner()
e0, grad0 = mf_grad_scan('N; N 1 1.2')
e1, grad1 = mf_grad_scan('N; N 1 1.199')
e2, grad2 = mf_grad_scan('N; N 1 1.201')
e_diff = (e2 - e1) / 0.002 * 0.529
print('finite difference', e_diff, 'analytical gradients', grad0[1,0])

mf_grad_scan.mol.unit = 'Bohr'
e0, grad0 = mf_grad_scan('N; N 1 1.8')
e1, grad1 = mf_grad_scan('N; N 1 1.799')
e2, grad2 = mf_grad_scan('N; N 1 1.801')
e_diff = (e2 - e1) / 0.002
print('finite difference', e_diff, 'analytical gradients', grad0[1,0])


# The gradients scanner can be pass to pyberny geometry optimizer
# See also examples/geomopt/01-pyberny.py and
# examples/geomopt/02-as_pyscf_method.py
from pyscf.geomopt import berny_solver
berny_solver.optimize(mf_grad_scan)

