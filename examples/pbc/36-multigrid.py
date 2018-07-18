#!/usr/bin/env python

'''
Use multigrid to accelerate DFT numerical integration.
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

kpts = cell.make_kpts([4,4,4])
mf = dft.KRKS(cell, kpts)
mf.xc = 'lda,vwn'

multigrid.multigrid(mf).kernel()

