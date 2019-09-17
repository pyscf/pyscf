#!/usr/bin/env python

'''
Different ways to input density fitting auxiliary basis for PBC Gaussian
density fitting.

See also pyscf/examples/df/01-auxbasis.py for more auxiliary basis input
methods.
'''

import numpy as np
from pyscf import gto as mol_gto
from pyscf.pbc import gto, scf, cc, df

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-dzv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

#
# Default DF auxiliary basis is a set of even-tempered gaussian basis (with
# exponents alpha * beta**i, i = 1,..,N).  The even-tempered parameter alpha
# is determined automatically based on the orbital basis.  beta is set to 2.0
#
mf = scf.RHF(cell).density_fit()
mf.kernel()


#
# By assigning the argument auxbasis of density_fit method, the DF calculation
# can use the assigned auxiliary basis. For example, in the statement below,
# the auxiliary basis is weigend Coulomb fitting basis augmented with one set
# of polarized d functions.
#
auxbasis = {'C': ('weigend', [[2, (0.5, 1)]])}
mf = scf.RHF(cell).density_fit(auxbasis=auxbasis)
mf.kernel()


#
# df.aug_etb is a shortcut function to create even-tempered gaussian basis
# based on the orbital basis.  The following example produces a dense etb
# basis spectrum.
#
auxbasis = df.aug_etb(cell, beta=1.6)
mf = scf.RHF(cell).density_fit(auxbasis=auxbasis)
mf.kernel()


#
# Another straightforward way to input even-tempered Gaussian basis (for
# better resolution of auxiliary basis) is the use of function gto.expand_etbs.
#
# Note the PBC Gaussian DF module will automatically remove diffused Gaussian
# fitting functions. It is controlled by the parameter eta.  The default value is
# 0.2 which removes all Gaussians whose exponents are smaller than 0.2.
# When your input DF basis has diffused functions, you need to reduce the
# value of  mf.with_df.eta  to reserve the diffused functions.  However,
# keeping diffused functions occasionally lead numerical noise in the GDF
# method.
#
auxbasis = {'C':
                               # (l, N, alpha, beta)
            mol_gto.expand_etbs([(0, 25, 0.15, 1.6), # 25s
                                 (1, 20, 0.15, 1.6), # 20p
                                 (2, 10, 0.15, 1.6), # 10d
                                 (3, 5 , 0.15, 1.6), # 5f
                                ])
           }
mf = scf.RHF(cell).density_fit(auxbasis=auxbasis)
mf.with_df.eta = 0.1
mf.kernel()


#
# The 3-index density fitting tensor can be loaded from the _cderi file.
# Using the 3-index tensor, the 4-center integrals can be constructed:
#    (pq|rs) = \sum_L A_lpq A_lrs
#
# The 3-index tensor for gamma point can be accessed with the code snippet
# below.  Assuming in the first pass, the GDF 3-index tensors are saved with
# the following code
#
mf = scf.RHF(cell, cell.make_kpts([2,2,2])).density_fit(auxbasis=auxbasis)
mf.with_df._cderi_to_save = 'pbc_gdf.h5'
mf.kernel()

#
# In the second pass, the GDF 3-index tensor can be accessed with a GDF
# object.
#
a_gdf = df.GDF(cell)
a_gdf._cderi = 'pbc_gdf.h5'
naux = a_gdf.get_naoaux()
nao = cell.nao
A_lpq = np.empty((naux,nao,nao))
kpt = np.zeros(3)
p1 = 0
for LpqR, LpqI in a_gdf.sr_loop((kpt,kpt), compact=False):
    p0, p1 = p1, p1 + LpqR.shape[0]
    A_lpq[p0:p1] = LpqR.reshape(-1,nao,nao)

from pyscf import lib
eri = lib.einsum('lpq,lrs->pqrs', A_lpq, A_lrs)

