#!/usr/bin/env python

import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, cell_from_ase
from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import UnitCellFilter, StrainFilter

atoms = bulk('Si', 'diamond', a=5.43)
# Only .atoms and .a are defined in this cell instance. It's necessary to assign
# basis set and pseudo potential
cell = cell_from_ase(atoms)
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
# Create a PySCF method, which will be used as a template in the following ASE
# calculator. Parameters (such as convergence tol) can be defined in the
# template.
mf = cell.KRKS(xc='pbe', kpts=kpts)
mf.conv_tol = 1e-6

# PySCF(method=...) creates an ASE calculator for the specified PySCF method.
# All methods implemented in PySCF can be used as templates to create a Calculator.
# However, some methods may not support gradients or stress tensors. Geometry
# optimization or lattice optimization that rely on these functionalities will
# be halted.
atoms.calc = PySCF(method=mf)

# optimize atom positions
opt = BFGS(atoms, logfile='atom_pos.log')
opt.run()
print(atoms.get_positions())


# Optimize both lattice and atom positions
opt = BFGS(UnitCellFilter(atoms), logfile='lattice_atom.log')
opt.run()
print(atoms.get_positions())
print(atoms.cell)


# Optimize lattice only. Atom positions (fractional coordinates) are frozen.
opt = BFGS(StrainFilter(atoms), logfile='lattice_atom.log')
opt.run()
print(atoms.cell)
