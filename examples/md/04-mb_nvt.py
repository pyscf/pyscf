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
myintegrator = pyscf.md.NVT(myhf, dt=5, steps=10, 
			     T=300, taut=50, veloc=init_veloc,
			     energy_output="NVT.md.energies", temp_output="NVT.md.temp", 
			     trajectory_output="NVT.md.xyz").run()
			     
# Close the file streams for the energy, temperature and trajectory.
myintegrator.energy_output.close()
myintegrator.temp_output.close()
myintegrator.trajectory_output.close()
