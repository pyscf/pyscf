#!/usr/bin/env python

'''
Gamma point Hartree-Fock/DFT for all-electron calculation

See also
examples/df/00-with_df.py
examples/df/01-auxbasis.py
examples/df/40-precomupte_df_ints.py
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

mf = scf.RHF(cell).density_fit()
mf.kernel()

# Or use even-tempered Gaussian basis as auxiliary fitting functions.
# The following auxbasis is generated based on the expression
#    alpha = a * 1.7^i   i = 0..N
# where a and N are determined by the smallest and largest exponents of AO basis.
import pyscf.df
auxbasis = pyscf.df.aug_etb(cell, beta=1.7)
mf = scf.RHF(cell).density_fit(auxbasis=auxbasis)
mf.kernel()

#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation
#
mf = dft.RKS(cell).density_fit(auxbasis='weigend')
mf.xc = 'bp86'
# You should first set mf.xc then apply newton method (see also
# examples/scf/22-newton.py)
mf = mf.newton()
mf.kernel()

#
# The computational costs to initialize PBC DF object is high.  The density
# fitting integral tensor created in the initialization can be cached for
# future use.  See also examples/df/40-precomupte_df_ints.py
#
mf = dft.RKS(cell).density_fit(auxbasis='weigend')
mf.with_df._cderi_to_save = 'df_ints.h5'
mf.kernel()
#
# The DF integral tensor can be preloaded in an independent calculation.
#
mf = dft.RKS(cell).density_fit()
mf.with_df._cderi = 'df_ints.h5'
mf.kernel()
