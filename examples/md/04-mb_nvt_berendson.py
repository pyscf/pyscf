#!/usr/bin/env python

'''
Run an NVT BOMD simulation using initial velocity from 
a Maxwell-Boltzmann distribution at 300K. 
'''

import pyscf
import pyscf.md

mol = pyscf.M(
    atom='''O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587''',
    basis = 'ccpvdz')

myhf = mol.RHF()

# initial velocities from a Maxwell-Boltzmann distribution [T in K and velocities are returned in (Bohr/ time a.u.)]
init_veloc = pyscf.md.distributions.MaxwellBoltzmannVelocity(mol, T=300)

# We set the initial velocity by passing to "veloc", 
#T is the ensemble temperature in K and taut is the Berendsen Thermostat time constant given in time a.u.
myintegrator = pyscf.md.integrators.NVTBerendson(myhf, dt=5, steps=100, 
			     T=300, taut=50, veloc=init_veloc,
			     data_output="NVT.md.data", 
			     trajectory_output="NVT.md.xyz").run()
			     
# Close the file streams for the energy, temperature and trajectory.
myintegrator.data_output.close()
myintegrator.trajectory_output.close()
