#!/usr/bin/env python

'''
In the pure DFT (LDA, GGA) calculations, Coulomb matrix of density fitting
method by default is computed on the fly (without caching the 3-index integral
tensor on disk).  For small systems, this is slightly slower than precomputing
the 3-index integral tensor.  On-the-fly Coulomb builder can be disabled by
explicitly constructing and caching the 3-index tensor.
'''

import time
from pyscf import gto, dft

mol = gto.Mole()
mol.atom = '''
O  1.081302  1.129735  1.195158
O  -.967942  1.693585   .543683
N  2.060859  1.075277 -1.315237
C   .249391  1.494424   .336070
C   .760991  1.733681 -1.081882
H  2.396597  1.201189 -2.305828
H  2.790965  1.427758  -.669398
H  1.985133   .067145 -1.148141
H   .883860  2.805965 -1.234913
H   .041439  1.369111 -1.813528
'''
mol.basis = 'ccpvdz'
mol.build()

#
# Default treatment for Coulomb matrix is IO free.
#
t0 = time.time()
mf = dft.RKS(mol).density_fit()
mf.kernel()
print('CPU time', time.time() - t0)
print(mf.with_df._cderi is None)

#
# Explicitly build and cache the 3-index tensor is slightly faster for small
# systems.  Since 3-index tensor is cached on disk, IO overhead may not be
# ignored for large systems.  Especially on multi-core machines, the IO free
# treatment above can be more efficient.
#
t0 = time.time()
mf = dft.RKS(mol).density_fit()
mf.with_df.build()
mf.kernel()
print('CPU time', time.time() - t0)
print(mf.with_df._cderi is None)  # ._cderi will not be created
