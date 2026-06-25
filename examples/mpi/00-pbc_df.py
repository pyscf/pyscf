#!/usr/bin/env python

'''
MPI parallelization with mpi4pyscf

https://github.com/sunqm/mpi4pyscf

mpi4pyscf allows you switching to MPI mode seamlessly by replacing certain
object, eg the density fitting object in the PBC calculations.
'''

import sys
import numpy
from pyscf.pbc import gto, scf, dft

verify_windows = '--pyscf-verify-windows' in sys.argv
try:
    from mpi4pyscf.pbc import df as mpidf
except ModuleNotFoundError:
    if verify_windows:
        # mpi4pyscf is an optional add-on for MPI examples.
        print('Skipping MPI density-fitting example during Windows verification because mpi4pyscf is not installed.')
        raise SystemExit(0)
    raise

cell = gto.M(
    h = numpy.eye(3)*3.5668,
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
    mesh = [20]*3,
    verbose = 4,
)

nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)

kmf = scf.KRHF(cell, kpts)
kmf.with_df = mpidf.FFTDF(cell, kpts)
kmf.kernel()

kmf = dft.KRKS(cell, kpts)
# Turn to the atomic grids if you like
kmf.grids = dft.gen_grid.BeckeGrids(cell)
kmf.xc = 'm06,m06'
kmf.with_df = mpidf.FFTDF(cell, kpts)
kmf.kernel()

