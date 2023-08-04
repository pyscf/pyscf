#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Example for the value of FCI wave function on specific electrons coordinates
'''

import numpy as np
import pyscf
from pyscf import fci

mol = pyscf.M(atom='H 0 0 0; H 0 0 1.1', basis='6-31g', verbose=0)
mf = mol.RHF().run()
e1, ci1 = fci.FCI(mol, mf.mo_coeff).kernel()
print('FCI energy', e1)

# coordinates for all electrons in the molecule
e_coords = np.random.rand(mol.nelectron, 3)
ao_on_grid = mol.eval_gto('GTOval', e_coords)
mo_on_grid = ao_on_grid.dot(mf.mo_coeff)

# View mo_on_grid as a transformation from MO to grid basis
mo_grid_ovlp = mo_on_grid.T

# ci_on_grid gives the value on e_grids for each CI determinant
ci_on_grid = fci.addons.transform_ci(ci1, mol.nelec, mo_grid_ovlp)
print(f'For electrons on grids\n{e_coords}\nCI value = {ci_on_grid.sum()}')
