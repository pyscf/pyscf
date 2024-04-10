#!/usr/bin/env python

'''
Samples the initial velocity for ethlyene using a Maxwell-Boltzmann
distribution at 300K. Run an NVE BOMD simulation using this initial velocity.
'''

import pyscf
import pyscf.md as md

# Ethylene molecule
mol = pyscf.M(verbose=3,
                     atom=''' C -0.0110224 -0.01183 -0.0271398
        C -0.000902273 0.0348566 1.34708
        H 1.07646 0.0030022 -0.456854
        H 0.976273 -0.140089 1.93039
        H -0.926855 -0.147441 1.98255
        H -0.983897 0.0103535 -0.538589
        ''',
                     unit='ANG',
                     basis='ccpvdz',
                     spin=0)

# We grab the initial velocities from a Maxwell-Boltzmann distribution
# The T parameter controls the temperature. Velocities are returned in a.u.
init_veloc = md.distributions.MaxwellBoltzmannVelocity(mol, T=300)

# Note that we can set the seed for the random number generator through the
# `md.set_seed()` function as
#
# >>> md.set_seed(1234)
#
# This allows for reproducible results.

# Prepare how to electronic structure method to propagate nuclei.
# Here we use HF
myhf = mol.RHF()

# We set the initial velocity by passing to "veloc"
myintegrator = pyscf.md.NVE(myhf, dt=5, steps=10, veloc=init_veloc, data_output="NVE.md.data")

myintegrator.run()

# We can also set it via
#
# >>> myintegrator.veloc = init_veloc
#
# or through the run/kernel function
#
# >>> myintegrator.kernel(veloc=init_veloc)
