#!/usr/bin/env python

from pyscf import gto
from pyscf import dft

'''
Tune DFT grids

By default, the DFT integration grids use
* Bragg radius for atom
* Treutler-Ahlrichs radial grids
* Becke partition for grid weights
* NWChem pruning scheme
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

# * Adjust Radial grids
# By overwriting the radi_method attribute, the radial grids can be generated
# using other schemes. The available settings include Treutler-Ahlrichs
# (default), Delley, Mura-Knowles, and Gauss-Chebyshev scheme.
#grids.radi_method = dft.treutler_ahlrichs
#grids.radi_method = dft.delley
#grids.radi_method = dft.mura_knowles
#grids.radi_method = dft.gauss_chebeshev
# See pyscf/dft/radi.py for more radial grid schemes
method = dft.RKS(mol)
method.grids.radi_method = dft.mura_knowles
print('Changed radial grids for DFT.  E = %.12f' % method.kernel())

# * Adjust weights assignment for each grid.
# The Becke partition scheme is utilized by default for grid weights.
# Another available option is Stratmann-Scuseria weight scheme
#method.grids.becke_scheme = dft.original_becke
# See pyscf/dft/gen_grid.py for detail of the grid weight scheme
method = dft.RKS(mol)
method.grids.becke_scheme = dft.stratmann
print('Changed grid partition function.  E = %.12f' % method.kernel())

# * Radial and angular grids settings.
# * The "Grids.level" attribute controls the radial and angular grids with
# pre-defined configurations.
# Grids level 0 - 9.  Big number indicates dense grids. Default is 3
method = dft.RKS(mol)
method.grids.level = 4
print('Dense grids.  E = %.12f' % method.kernel())

# By assigning a dictionary to the "atom_grid" attribute, you can customize the 
# grid settings for specific atoms. For atoms in mol.atom that are not included 
# in the Grids.atom_grid attribute, the grid settings associated with
# Grids.level will be applied.
method = dft.RKS(mol)
method.grids.atom_grid = {'O': (100, 770)}
print('Dense grids for O atom.  E = %.12f' % method.kernel())

# * Specifying mesh grids for all atoms.
# You can assign a tuple directly to the Grids.atom_grid attribute to apply the 
# same grid settings to all atoms.
method = dft.RKS(mol)
method.grids.atom_grid = (100, 770)
print('Dense grids for all atoms.  E = %.12f' % method.kernel())

# * Assigning default mesh settings, with exceptions specified as key-value pairs 
# in the "Grids.atom_grid" dictionary.
method.grids.atom_grid = {'default': (40, 302), 'O': (100, 770)}

# Disable pruning grids near core region
#grids.prune = dft.sg1_prune
method = dft.RKS(mol)
method.grids.prune = None
print('Changed grid partition function.  E = %.12f' % method.kernel())
