#!/usr/bin/env python
#
# Author: Narbe Mardirossian <nmardirossian@berkeley.edu>
#

'''
A simple example to run density functional with non-local correlation calculation.

Available NLC functionals: wB97M_V, wB97X_V, B97M_V
'''

from pyscf import gto, dft

mol = gto.M(atom='H    0.000000000  -0.120407870  -0.490828400; F    0.000000000   0.009769450  -1.404249780', basis='6-31G', symmetry=False, verbose=10, unit='Angstrom', spin=0)
mf = dft.RKS(mol)
mf.xc='wB97M_V'
mf.nlc='VV10'

mf.grids.atom_grid={'H': (99,590),'F': (99,590)}
mf.grids.prune=None

mf.nlcgrids.atom_grid={'H': (50,194),'F': (50,194)}
mf.nlcgrids.prune=dft.gen_grid.sg1_prune

mf.kernel()
