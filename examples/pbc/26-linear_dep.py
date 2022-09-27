#!/usr/bin/env python

'''
Removing linear dependence AO functions in PBC calculations.

See also pyscf/examples/scf/11-linear_dep.py
'''

from pyscf import scf as mol_scf
from pyscf.pbc import gto, dft

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-qzvp'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

mf = dft.KRKS(cell, cell.make_kpts([2,2,2]))
mf = mol_scf.addons.remove_linear_dep_(mf)
mf.kernel()
