#!/usr/bin/env python

from pyscf import gto
from pyscf import dft

'''
Tune DFT grids

By default, the DFT grid employs
* Bragg radius for atom
* Treutler-Ahlrichs radial grids
* Becke partition for grid weight
* Treutler-Ahlrichs pruning scheme
* mesh grids
  ===================================
  Elements  radial part  angular part
  --------  -----------  ------------
  H, He         50           302
  Li - Ne       75           302
  Na - Ar       80           434
  K  - Kr       90           434
  Rb - Xe       95           434
  Cs - Rn      100           434
  ===================================

See pyscf/dft/gen_grid.py  "class Grids" for more details.
'''

mol = gto.M(
    verbose = 0,
    atom = '''
    o    0    0.       0.
    h    0    -0.757   0.587
    h    0    0.757    0.587''',
    basis = '6-31g')
method = dft.RKS(mol)
print('Default DFT(LDA).  E = %.12f' % method.kernel())

# See pyscf/dft/radi.py for more radial grid schemes
#grids.radi_method = dft.radi.gauss_chebeshev
#grids.radi_method = dft.radi.delley
method = dft.RKS(mol)
method.grids.radi_method = dft.radi.mura_knowles
print('Changed radial grids for DFT.  E = %.12f' % method.kernel())


# See pyscf/dft/gen_grid.py for detail of the grid weight scheme
#method.grids.becke_scheme = dft.gen_grid.original_becke
# Stratmann-Scuseria weight scheme
method = dft.RKS(mol)
method.grids.becke_scheme = dft.gen_grid.stratmann
print('Changed grid partition funciton.  E = %.12f' % method.kernel())

# Grids dense level 0 - 6.  Big number indicates dense grids. Default is 3
method = dft.RKS(mol)
method.grids.level = 4
print('Dense grids.  E = %.12f' % method.kernel())

# Specify mesh grid for certain atom
method = dft.RKS(mol)
method.grids.atom_grid = {'O': (100, 770)}
print('Dense grids for O atom.  E = %.12f' % method.kernel())

