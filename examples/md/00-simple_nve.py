#!/usr/bin/env python

'''
A simple example to run an NVE BOMD simulation.
'''

import pyscf
import pyscf.md

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

myhf = mol.RHF().run()

# 6 orbitals, 8 electrons
mycas = myhf.CASSCF(6, 8)
myscanner = mycas.nuc_grad_method().as_scanner()

# Generate the integrator
# sets the time step to 5 a.u. and will run for 100 steps
# or for 50 a.u.
myintegrator = pyscf.md.NVE(myscanner,
                            dt=5,
                            steps=10,
                            data_output="BOMD.md.data",
                            trajectory_output="BOMD.md.xyz").run()

# Note that we can also just pass the CASSCF object directly to
# generate the integrator and it will automatically convert it to a scanner
# myintegrator = pyscf.md.NVE(mycas, dt=5, steps=100)

# Close the file streams for the energy and trajectory.
myintegrator.data_output.close()
myintegrator.trajectory_output.close()
