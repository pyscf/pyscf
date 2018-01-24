"""
Take ASE Diamond structure, input into PySCF and run
"""

import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
from pyscf.pbc.tools import pyscf_ase

import ase
import ase.lattice
from ase.lattice.cubic import Diamond


ase_atom=Diamond(symbol='C', latticeconstant=3.5668)
print(ase_atom.get_volume())

cell = pbcgto.Cell()
cell.verbose = 5
cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a=ase_atom.cell
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.build()

mf=pbcdft.RKS(cell)

mf.xc='lda,vwn'

print(mf.kernel()) # [10,10,10]: -44.8811199336

