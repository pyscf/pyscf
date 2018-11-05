#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Analytical nuclear gradients can be calculated with the background charges.
The nuclear gradients driver (nuc_grad_method) can be called the same way as
the regular calculations.

Note:
1. the mcscf nuclear gradients have to be calculated with the (recommended)
   first initialization method. See also example 02-mcscf.py
2. X2C gradients with QM/MM charges are not supported.
'''

import numpy
from pyscf import gto, scf, ci, mcscf, tddft, qmmm

mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
            ''',
            basis='3-21g')

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * -.1

mf = qmmm.mm_charge(scf.RHF(mol), coords, charges).run()

mf.nuc_grad_method().run()

ci.CISD(mf).run().nuc_grad_method().run()

mc = mcscf.CASCI(mf, 6, 6).run()
mc.nuc_grad_method().run()

tddft.TDA(mf).run().nuc_grad_method().run(state=2)
