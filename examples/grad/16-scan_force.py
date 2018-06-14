#!/usr/bin/env python

'''
Scan molecule dissociation curve and the force on the curve.
'''

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, dft

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

plt.plot(bond, e_hf[::-1])
plt.show()

plt.plot(bond, force[::-1])
plt.show()

