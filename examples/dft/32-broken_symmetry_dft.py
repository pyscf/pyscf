#!/usr/bin/env python

'''
Broken-symmetry DFT 
'''

import numpy
from pyscf import gto
from pyscf import dft

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
Fe     5.22000000    1.05000000   -7.95000000
Fe     5.88000000   -1.05000000   -9.49000000
S      3.86000000   -0.28000000   -9.06000000
S      7.23000000    0.28000000   -8.38000000
'''
mol.basis = 'dzp'
mol.spin = 8
mol.build()

#
# First converge a high-spin UKS calculation
#
mf = dft.UKS(mol)
mf.xc = 'bp86'
mf.level_shift = 0.1
mf.conv_tol = 1e-4
mf.kernel()

#
# Flip the local spin of the first Fe atom ('0 Fe' in ao_labels)
#
idx_fe1 = mol.search_ao_label('0 Fe')
dma, dmb = mf.make_rdm1()
dma_fe1 = dma[idx_fe1.reshape(-1,1),idx_fe1].copy()
dmb_fe1 = dmb[idx_fe1.reshape(-1,1),idx_fe1].copy()
dma[idx_fe1.reshape(-1,1),idx_fe1] = dmb_fe1
dmb[idx_fe1.reshape(-1,1),idx_fe1] = dma_fe1
dm = [dma, dmb]

#
# Change the spin and run the second pass for low-spin solution
#
mol.spin = 0
mf = dft.UKS(mol)
mf.xc = 'bp86'
# Apply large level shift to avoid big oscillation at the beginning of SCF
# iteration.  This is not a must for BS-DFT.  Depending on the system, this
# step can be omitted.
mf.level_shift = 1.0
mf.conv_tol = 1e-4
mf.kernel(dm)

#
# Remove the level shift and converge the low-spin state in the final pass
#
mf.level_shift = 0
mf.conv_tol = 1e-9
mf.kernel(mf.make_rdm1())

