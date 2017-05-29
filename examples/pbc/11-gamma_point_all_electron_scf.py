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

mf = dft.RKS(cell).mix_density_fit(auxbasis='weigend')
mf.xc = 'bp86'
mf.kernel()

#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation
#
mf = scf.RHF(cell).mix_density_fit(auxbasis='weigend')
mf = scf.newton(mf)
mf.kernel()

