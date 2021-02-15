# # Easy Ab initio calculation with ASE-Siesta-Pyscf
# ## No installation necessary, just download a ready to go container for any system, or run it into the cloud

# import libraries and set up the molecule geometry

from ase.units import Ry, eV, Ha
from ase.calculators.siesta import Siesta
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt

H2O = Atoms('H2O', positions = [[-0.757,  0.586,  0.000],
                                [0.757,  0.586,  0.000],
                                [0.0, 0.0, 0.0]],
            cell=[20, 20, 20])

# enter siesta input and run siesta
siesta = Siesta(
    mesh_cutoff=150 * Ry,
    basis_set='DZP',
    pseudo_qualifier='lda',
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

H2O.set_calculator(siesta)
e = H2O.get_potential_energy()


# ### The TDDFT calculations with PySCF-NAO

siesta.pyscf_tddft(label="siesta", jcutoff=7, iter_broadening=0.15/Ha,
        xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7, freq = np.arange(0.0, 15.0, 0.05))

# plot polarizability with matplotlib

fig = plt.figure(1)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(siesta.results["freq range"], siesta.results["polarizability nonin"][:, 0, 0].imag)
ax2.plot(siesta.results["freq range"], siesta.results["polarizability inter"][:, 0, 0].imag)

ax1.set_xlabel(r"$\omega$ (eV)")
ax2.set_xlabel(r"$\omega$ (eV)")

ax1.set_ylabel(r"Im($P_{xx}$) (au)")
ax2.set_ylabel(r"Im($P_{xx}$) (au)")

ax1.set_title(r"Non interacting")
ax2.set_title(r"Interacting")

fig.tight_layout()
plt.show()
