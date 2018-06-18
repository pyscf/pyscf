#!/usr/bin/env python

'''
Input ke_cutoff (kinetic energy cutoff) or mesh (#grids) for FFT-based Coulomb
integral.

If ke_cutoff and mesh are not specified in the input, they will be chosen
automatically in cell.build function based on the basis set.  You can set
ke_cutoff or mesh to control the performance/accuracy of Coulomb integrals.
'''

import numpy
import pyscf.lib
from pyscf.pbc import gto

cell = gto.Cell()
cell.atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = numpy.eye(3)*3.5668

cell.mesh = [25,25,25]  # 25 grids on each direction, => 25^3 grids in total
cell.build()

cell.ke_cutoff = 40 # Eh ~ mesh = [20,20,20] ~ 21^3 grids in total
cell.build()
