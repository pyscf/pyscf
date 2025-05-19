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
# Convert mol back to cell using the to_cell function.
# Lattice vectors "a" are not available in the mole object. Specify a "box" to
# place the mole in the cell.
#
cell_0D = mol.to_cell(box=a, dimension=0)
mf = pbcscf.RHF(cell).density_fit().run()
print('E(HF) with 0D PBC RHF calculation %s' % mf.e_tot)

#
# By transforming Mole to Cell instance, we can apply the MultiGrid integral
# algorithm for DFT calculations to fastly evaluate the Coulomb and XC functional.
#
cell_0D.verbose = 5
mf = cell_0D.RKS(xc='pbe')
mf.run() # This calls the standard numint module
mf = mf.multigrid_numint()
mf.run() # This calls the MultiGridNumInt algorithm
