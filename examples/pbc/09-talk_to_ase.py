"""
Take ASE structure, PySCF object,
and run through ASE calculator interface.

This allows other ASE methods to be used with PySCF;
here we try to compute an equation of state.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf.pbc.tools import pyscf_ase

import ase
import ase.lattice
from ase.build import bulk
# from ase.lattice.cubic import Diamond
from ase.units import kJ
from ase.eos import EquationOfState


ase_atom = bulk('C', 'diamond', a=3.5668)

# cell_from_ase function sets up a cell with cell.atom and cell.a initialized
# from ASE atoms. Everything else for a PySCF calculation should be specified to
# the cell.
cell = pyscf_ase.cell_from_ase(ase_atom)
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

# Set up a template calculation, which will be used for the ASE calculator.
# Additional variables can be assigned to the template method.
# E.g. SCF with k-point sampling can be set to
mf = cell.KRKS(xc='pbe', kpts=cell.make_kpts([2,2,2])).density_fit()

# Once this is setup, ASE is used for everything from this point on
ase_atom.calc = pyscf_ase.PySCF(method=mf)

print("ASE energy", ase_atom.get_potential_energy())
print("ASE energy (should avoid re-evaluation)", ase_atom.get_potential_energy())

# Plot band structure and save to figure C-bands.png
bs = ase_atom.calc.band_structure()
ax = bs.plot(filename='C-bands.png', emax=20, emin=-20)
plt.close(ax.figure)

# Compute density of states using ASE's DOS module
from ase.dft.dos import DOS

# Gaussian smearing DOS
dos_gauss = DOS(ase_atom.calc, width=0.1, window=(-20, 20), npts=1000)
d_gauss = dos_gauss.get_dos()

# Tetrahedron method DOS (width=0.0)
dos_tetra = DOS(ase_atom.calc, width=0.0, window=(-20, 20), npts=1000)
d_tetra = dos_tetra.get_dos()

# Plot DOS
def plot_dos(d_gauss, d_tetra, energies):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energies, d_gauss, label='Gaussian (0.1 eV)')
    ax.plot(energies, d_tetra, label='Tetrahedron')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('DOS')
    ax.set_title('Density of States')
    ax.legend()
    fig.savefig('dos.png')
    plt.close(fig)

plot_dos(d_gauss, d_tetra, dos_gauss.energies)

# Compute equation of state
ase_cell=ase_atom.cell
volumes = []
energies = []
for x in np.linspace(0.95, 1.15, 5):
    ase_atom.set_cell(ase_cell * x, scale_atoms = True)
    print("[x: %f, E: %f]" % (x, ase_atom.get_potential_energy()))
    volumes.append(ase_atom.get_volume())
    energies.append(ase_atom.get_potential_energy())

eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print(B / kJ * 1.0e24, 'GPa')
eos.plot('eos.png')
