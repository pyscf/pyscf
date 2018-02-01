#!/usr/bin/env python

'''
Hartree-Fock/DFT with k-points sampling for all-electron calculation
MDF (mixed density fitting) can also be used in k-points sampling.
'''

import numpy
from pyscf.pbc import gto, scf, dft

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = '6-31g',
    verbose = 4,
)

nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)

kmf = scf.KRHF(cell, kpts).mix_density_fit()
# In the MDF scheme, modifying the default mesh for PWs to reduce the cost
# The default mesh for PWs is a very dense-grid scheme which is automatically
# generated based on the AO basis. It is often not necessary to use dense grid
# for MDF method.
kmf.with_df.mesh = [10,10,10]
kmf.kernel()

kmf = dft.KRKS(cell, kpts).density_fit(auxbasis='weigend')
kmf.xc = 'bp86'
kmf.kernel()

#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation.  Note second order SCF algorithm does not support
# smearing method.
#
mf = scf.KRHF(cell, kpts).density_fit()
mf = mf.newton()
mf.kernel()

