"""
Take ASE structure, PySCF object,
and run through ASE calculator interface. 

This allows other ASE methods to be used with PySCF;
here we try to compute an equation of state.
"""

import numpy as np
from pyscf.pbc.tools import pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

import ase
import ase.lattice
from ase.lattice.cubic import Diamond
from ase.units import kJ
from ase.eos import EquationOfState


ase_atom=Diamond(symbol='C', latticeconstant=3.5668)

# cell_from_ase function sets up a cell with cell.atom and cell.a initialized
# from ASE atoms. Everything else for a PySCF calculation should be specified to
# the cell.
cell = cell_from_ase(ase_atom)
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 0

# Set up a template calculation, which will be used for the ASE calculator.
# Additional variables can be assigned to the template method.
# E.g. SCF with k-point sampling can be set to
mf = cell.KRKS(xc='lda,vwn', kpts=cell.make_kpts([2,2,2]))

# Once this is setup, ASE is used for everything from this point on
ase_atom.calc = pyscf_ase.PySCF(method=mf)

print("ASE energy", ase_atom.get_potential_energy())
print("ASE energy (should avoid re-evaluation)", ase_atom.get_potential_energy())
# Compute equation of state
ase_cell=ase_atom.cell
volumes = []
energies = []
for x in np.linspace(0.95, 1.2, 5):
    ase_atom.set_cell(ase_cell * x, scale_atoms = True)
    print("[x: %f, E: %f]" % (x, ase_atom.get_potential_energy()))
    volumes.append(ase_atom.get_volume())
    energies.append(ase_atom.get_potential_energy())

eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print(B / kJ * 1.0e24, 'GPa')
eos.plot('eos.png')
