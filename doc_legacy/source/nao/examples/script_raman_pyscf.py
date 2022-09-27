"""Example, in order to run you must place a pseudopotential 'Na.psf' in
the folder"""

from ase.units import Ry, eV, Ha
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.siesta_raman import SiestaRaman
from ase import Atoms
import numpy as np

# Define the systems
# example of Raman calculation for CO2 molecule,
# comparison with QE calculation can be done from
# https://github.com/maxhutch/quantum-espresso/blob/master/PHonon/examples/example15/README

CO2 = Atoms('CO2',
            positions=[[-0.009026, -0.020241, 0.026760],
                       [1.167544, 0.012723, 0.071808],
                       [-1.185592, -0.053316, -0.017945]],
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
        'XML.Write': True,
        'DM.UseSaveDM': True})

CO2.set_calculator(siesta)

ram = SiestaRaman(CO2, siesta, label="siesta", jcutoff=7, iter_broadening=0.15/Ha,
        xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7, freq = np.arange(0.0, 5.0, 0.05))
ram.run()
ram.summary(intensity_unit_ram='A^4 amu^-1')

ram.write_spectra(start=200)
