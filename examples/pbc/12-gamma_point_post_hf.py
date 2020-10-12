#!/usr/bin/env python

'''
Gamma point post-HF calculation needs only real integrals.
Methods implemented in finite-size system can be directly used here without
any modification.
'''

import numpy
from pyscf.pbc import gto, scf

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
    basis = '6-31g',
    verbose = 4,
)

mf = scf.RHF(cell).density_fit()
mf.with_df.mesh = [10]*3
mf.kernel()

#
# Import CC, TDDFT module from the molecular implementations
#
from pyscf import cc, tddft
mycc = cc.CCSD(mf)
mycc.kernel()

mytd = tddft.TDHF(mf)
mytd.nstates = 5
mytd.kernel()
