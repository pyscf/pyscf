#!/usr/bin/env python

'''
Gamma point Hartree-Fock/DFT for all-electron calculation

The default FFT-based 2-electron integrals may not be accurate enough for
all-electron calculation.  It's recommended to use MDF (mixed density fitting)
technique to improve the accuracy.
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
    gs = [10]*3,
    verbose = 4,
)

mf = scf.RHF(cell).mix_density_fit(auxbasis='weigend')
mf.kernel()

# Or use even-tempered Gaussian basis as auxiliary fitting functions.
# The following auxbasis is generated based on the expression
#    alpha = a * 1.7^i   i = 0..N
# where a and N are determined by the smallest and largest exponets of AO basis.
import pyscf.df
auxbasis = pyscf.df.addons.aug_etb_for_dfbasis(cell, beta=1.7, start_at=0)
mf = dft.RKS(cell).mix_density_fit(auxbasis=auxbasis)
mf.xc = 'bp86'
mf.kernel()

#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation
#
mf = scf.RHF(cell).mix_density_fit(auxbasis='weigend')
mf = scf.newton(mf)
mf.kernel()

