#!/usr/bin/env python

'''
Different ways to input density fitting auxiliary basis for PBC Gaussian
density fitting.

See also pyscf/examples/df/01-auxbasis.py for more auxiliary basis input
methods.
'''

from pyscf.gto import mole
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
mf = scf.RHF(cell).density_fit(auxbasis=auxbasis)
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

