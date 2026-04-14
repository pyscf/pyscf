#!/usr/bin/env python

'''
Removing linear dependence AO functions in PBC calculations.

Linear dependency in the AO overlap matrix is controlled globally via
pyscf.scf.hf.remove_overlap_zero_eigenvalue
pyscf.scf.hf.overlap_zero_eigenvalue_threshold

These settings are also respected by PBC modules.
'''

from pyscf import scf as mol_scf
from pyscf.pbc import gto, dft

mol_scf.hf.remove_overlap_zero_eigenvalue = True
mol_scf.hf.overlap_zero_eigenvalue_threshold = 1e-6

aug_basis = [[0, [0.08, 1]], [0, [0.12, 1]]]

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
# Augment diffuse functions
cell.basis = ('gth-dzvp', aug_basis)
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

mf = dft.KRKS(cell, cell.make_kpts([2,2,2]))
mf.kernel()
