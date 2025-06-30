#!/usr/bin/env python

'''
Use multi-grid to accelerate DFT numerical integration.
'''

import numpy
from pyscf.pbc import gto, dft
from pyscf.pbc.dft import multigrid

cell = gto.M(
    verbose = 4,
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'sto3g',
    #basis = 'ccpvdz',
    #basis = 'gth-dzvp',
    #pseudo = 'gth-pade'
)

mf = dft.UKS(cell)
mf.xc = 'lda,vwn'

#
# Call the multigrid_numint() method to enable multigrid integration.
#
mf = mf.multigrid_numint()
mf.kernel()

#
# MultiGridFFTDF can be used with second order SCF solver.
#
mf = mf.newton()
mf.kernel()
