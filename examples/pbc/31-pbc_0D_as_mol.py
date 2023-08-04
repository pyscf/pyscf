#!/usr/bin/env python

'''
Convert back and forth between the molecule (open boundary) and the 0D PBC
system.
'''

import numpy
from pyscf import gto, scf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df

cell = pbcgto.Cell()
cell.atom = 'N 0 0 0; N 0 0 1.2'
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = numpy.eye(3)
cell.dimension = 0
cell.symmetry = True
cell.build()

mf = pbcscf.RHF(cell)
mf.with_df = df.AFTDF(cell)
mf.run()
print('E(HF) with 0D PBC RHF calculation %s' % mf.e_tot)

#
# Convert cell to mol.
#
# Except lattice vectors, the mole object inherits all parameters from the
# cell object, like geometry, basis sets, and pseudopotential.  Using the
# generated mol object with molecular code, it should produce the same results
# as the 0D PBC calculation
#
mol = cell.to_mol()
mf = scf.RHF(mol).run()
print('E(HF) with molecular RHF calculation %s' % mf.e_tot)

# Cell and Mole have almost the same structure. If cell was fed to the
# molecular functions, the code is able to handle the cell without any
# errors. However, due to the different treatments of nuclear repulsion
# energy, a small discrepancy will be found in the total energy.
mf = scf.RHF(cell).run()
print('E(HF) of molecular RHF with cell %s' % mf.e_tot)

#
# Convert mol back to cell.
#
# The mol ojbect above contains all information of the pbc system which was
# initialized at the beginning. Using the "view" method to convert mol back to
# the cell object, all information can be transfer to the resultant cell
# object. Lattice vectors "a" are not available in the mole object. It needs
# to be specified in the cell.
#
cell_0D = mol.view(pbcgto.Cell)
cell_0D.a = numpy.eye(3)
cell_0D.dimension = 0
mf = pbcscf.RHF(cell).density_fit().run()
print('E(HF) with 0D PBC RHF calculation %s' % mf.e_tot)
