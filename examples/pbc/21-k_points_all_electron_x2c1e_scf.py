#!/usr/bin/env python

'''
X2C1e Hartree-Fock/DFT with k-points sampling for all-electron calculations.
'''

import numpy as np
from pyscf.pbc import gto, scf, dft

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = '6-31g'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

nk = [2,2,2]  # 2 k-poins for each axis, 2^3=8 kpts in total
kpts = cell.make_kpts(nk)

#
# Spin-free x2c1e (sfx2c1e) HF/DFT
# Only the scalar relativistic effects are included.
#
mf = scf.KRHF(cell, kpts).density_fit().sfx2c1e()
mf.kernel()

mf = dft.KRKS(cell, kpts).density_fit().sfx2c1e()
mf.xc = 'lda'
mf.kernel()


#
# x2c1e HF/DFT
# Generalized SCF is necessary to address off-diagonal blocks in the spin space
#
mf = scf.KGHF(cell, kpts).density_fit().x2c1e()
mf.kernel()

mf = dft.KGKS(cell, kpts).density_fit().x2c1e()
mf.xc = 'lda'
mf.kernel()
