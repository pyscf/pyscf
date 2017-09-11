#!/usr/bin/env python

'''
Input ke_cutoff (kinetic energy cutoff) or gs (#grids) for FFT-based Coulomb
integral.

If ke_cutoff and gs are not specified in the input, they will be are chosen
automatically in cell.build function based on the basis set.  You can set
ke_cutoff or gs to control the performance/accuracy of Coulomb integrals.
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

cell.gs = [12,12,12]  # 12 grids on positive x direction, => 25^3 grids in total
cell.build()

cell.ke_cutoff = 40 # Eh ~ gs = [10,10,10] ~ 21^3 grids in total
cell.build()

