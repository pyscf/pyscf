"""Example, in order to run you must place a pseudopotential 'Na.psf' in
the folder"""

from ase.units import Ry, eV, Ha
from ase.calculators.siesta import Siesta
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt

# Define the systems
Na8 = Atoms('Na8',
            positions=[[-1.90503810, 1.56107288, 0.00000000],
                       [1.90503810, 1.56107288, 0.00000000],
                       [1.90503810, -1.56107288, 0.00000000],
                       [-1.90503810, -1.56107288, 0.00000000],
                       [0.00000000, 0.00000000, 2.08495836],
                       [0.00000000, 0.00000000, -2.08495836],
                       [0.00000000, 3.22798122, 2.08495836],
                       [0.00000000, 3.22798122, -2.08495836]],
            cell=[20, 20, 20])

# enter siesta input
siesta = Siesta(
    mesh_cutoff=150 * Ry,
    basis_set='DZP',
    pseudo_qualifier='',
    energy_shift=(10 * 10**-3) * eV,
    fdf_arguments={
        'SCFMustConverge': False,
        'COOP.Write': True,
        'WriteDenchar': True,
        'PAO.BasisType': 'split',
        'DM.Tolerance': 1e-4,
        'DM.MixingWeight': 0.01,
        'MaxSCFIterations': 300,
        'DM.NumberPulay': 4,
        'XML.Write': True})


Na8.set_calculator(siesta)
e = Na8.get_potential_energy()
freq, pol = siesta.get_polarizability_pyscf_inter(label="siesta", jcutoff=7, iter_broadening=0.15/Ha,
        xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7, freq = np.arange(0.0, 5.0, 0.05))

# plot polarizability
plt.plot(freq, pol[:, 0, 0].imag)

plt.show()
