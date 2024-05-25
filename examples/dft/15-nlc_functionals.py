#!/usr/bin/env python
#
# Author: Narbe Mardirossian <nmardirossian@berkeley.edu>
#

'''
A simple example to run density functional with non-local correlation calculation.
'''

from pyscf import gto, dft

mol = gto.M(atom='H    0.000000000  -0.120407870  -0.490828400; F    0.000000000   0.009769450  -1.404249780',
            basis='6-31G')
mf = dft.RKS(mol)
mf.xc='wB97M_V'
mf.nlc='VV10'

mf.grids.atom_grid={'H': (99,590),'F': (99,590)}
mf.grids.prune=None

mf.nlcgrids.atom_grid={'H': (50,194),'F': (50,194)}
mf.nlcgrids.prune=dft.gen_grid.sg1_prune

mf.kernel()

# If mf.nlc is not configured, NLC will be computed based on the .xc settings.
# VV10 will be computed in the following example.
mf = dft.RKS(mol)
mf.xc = 'wb97m_v'
mf.run()

# To disable NLC even if the .xc requires a NLC correction, you can explicitly
# set nlc to False (or 0). This configuration might be commonly required when D3 and D4
# corrections are applied to the DFT calcuation.
mf.xc = 'wb97m_v'
mf.nlc = 0
mf.kernel()
