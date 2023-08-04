#!/usr/bin/env python

'''
A simple example to run an NVE BOMD simulation.
'''

import pyscf
import pyscf.md

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    verbose = 3,
    basis = 'ccpvdz',
    spin = 2)

myhf = mol.RHF().run()

# 6 orbitals, 8 electrons
mycas = myhf.CASSCF(6, 8)
myscanner = mycas.nuc_grad_method().as_scanner()

# Generate the integrator
# sets the time step to 5 a.u. and will run for 10 steps
# or for 50 a.u.
myintegrator = pyscf.md.NVE(myscanner,
                            dt=5,
                            steps=10)


def my_callback(local_dict):
    local_dict['scanner'].base.verbose = 4
    local_dict['scanner'].base.analyze()
    local_dict['scanner'].base.verbose = 3

myintegrator.callback = my_callback

myintegrator.run()

