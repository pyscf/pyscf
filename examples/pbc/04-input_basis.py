#!/usr/bin/env python

'''
Basis can be input the same way as the finite-size system.
'''

#
# Note pbc.gto.parse does not support NWChem format.  To parse NWChem format
# basis string, you need the molecule gto.parse function.
#

import numpy
from pyscf import gto
from pyscf.pbc import gto as pgto
cell = pgto.M(
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = {'C': gto.parse('''
# Parse NWChem format basis string (see https://bse.pnl.gov/bse/portal).
# Comment lines are ignored
#BASIS SET: (6s,3p) -> [2s,1p]
O    S
    130.7093200              0.15432897       
     23.8088610              0.53532814       
      6.4436083              0.44463454       
O    SP
      5.0331513             -0.09996723             0.15591627       
      1.1695961              0.39951283             0.60768372       
      0.3803890              0.70011547             0.39195739       
                                ''')},
    pseudo = 'gth-pade',
    a = numpy.eye(3)*3.5668)
