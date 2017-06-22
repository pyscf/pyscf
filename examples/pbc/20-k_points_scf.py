#!/usr/bin/env python

'''
Mean field with k-points sampling

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.
'''

from pyscf.pbc import gto, scf, dft
import numpy

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
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    gs = [10]*3,
    verbose = 4,
)

nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)

kmf = scf.KRHF(cell, kpts)
kmf.kernel()

kmf = dft.KRKS(cell, kpts)
# Turn to the atomic grids if you like
kmf.grids = dft.gen_grid.BeckeGrids(cell)
kmf.xc = 'm06'
kmf.kernel()

#
# Second order SCF solver (which is defined in pbc module) should be used
# in the k-point calculations
#
mf = scf.newton(scf.KRHF(cell, kpts))
mf.kernel()

