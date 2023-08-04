#!/usr/bin/env python

import numpy
import pyscf.lib
from pyscf.pbc import gto

#
# Simliar to the initialization of "Mole" object, here we need create a "Cell"
# object for periodic boundary systems.
#
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
#
# Note the extra attribute ".a" in the "cell" initialization.
# .a is a matrix for lattice vectors.  Each row of .a is a primitive vector.
#
cell.a = numpy.eye(3)*3.5668
cell.build()

#
# pbc.gto module provided a shortcut initialization function "gto.M", like the
# one of finite size problem
#
cell = gto.M(
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
    a = numpy.eye(3)*3.5668)

