#!/usr/bin/env python

'''
Run the charged system with Makov-Payne correction.
'''

import numpy
from pyscf.pbc import gto, scf

cell = gto.M(atom='Al 0 0 0', basis='lanl2dz', ecp='lanl2dz',
             spin=0, a=numpy.eye(3)*6, charge=1, dimension=3)
mf = scf.RHF(cell)
mf.run()
