#!/usr/bin/env python

'''
Run an NVE BOMD simulation while supplying some initial velocity.
'''

import pyscf
import pyscf.md

mol = pyscf.M(
    atom='''O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587''',
    basis = 'ccpvdz')

myhf = mol.RHF()

# Set the initial velocity in a numpy array
import numpy as np
init_veloc = np.array([[0.000336, 0.000044, 0.000434],
                               [-0.000364, -0.000179, 0.001179],
                               [-0.001133, -0.000182, 0.000047]])
# This will supply the initial velocity for oxygen to be
# 0.000336, 0.000044, 0.000434
# the first hydrogen will then have an initial velocity of
# -0.000364, -0.000179, 0.001179
# and the last hydrogen will have an initial velocity of
# -0.001133, -0.000182, 0.000047
# The units are in atomic units (Bohr/ time a.u.)


# Generate the integrator
# sets the time step to 5 a.u. and will run for 100 steps
# or for 500 a.u.
myintegrator = pyscf.md.NVE(myhf, dt=5, steps=100, veloc=init_veloc).run()

# We can also construct the integrator first like
#
# myintegrator = pyscf.md.NVE(myhf, dt=5, steps=100)
#
# and then supply the velocity as
#
# myintegrator.veloc = init_veloc
#
# or we can pass it during the kernel/run function
#
# myintegrator.kernel(veloc=init_veloc)

