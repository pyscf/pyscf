#!/usr/bin/env python

'''
Access AO integrals in PBC code
'''


from pyscf.pbc import gto, df
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
    verbose = 4,
)

#
# 1-electron (periodic) integrals can be obtained with pbc_intor function.
# Note .intor function computes integrals without PBC
#
overlap = cell.pbc_intor('cint1e_ovlp_sph')
kinetic = cell.pbc_intor('cint1e_kin_sph')

#
# In PBC, 2e integrals are all computed with 3-ceter integrals, ie from
# density fitting module.  Without kpts argument, the DF gives gamma point AO
# 2e-integrals.  Permutation symmetry is not considered
#
mydf = df.FFTDF(cell)
eri = mydf.get_eri()
print('ERI shape (%d,%d)' % eri.shape)

kpts = numpy.array((
    (-.5, .5, .5),
    ( .5,-.5, .5),
    ( .5, .5,-.5),
    ( .5, .5, .5),
))
#
# Note certain translational symmetry is required:
#       kj-ki+kl-kk = 2n\pi
#
ki, kj, kk = kpts[:3]
kl = kk + ki - kj

mydf.kpts = kpts
eri = mydf.get_eri((ki,kj,kk,kl))
print('ERI shape (%d,%d)' % eri.shape)

#
# Use MDF method to get better accuracy.
#
mydf = df.MDF(cell, kpts)
mydf.auxbasis = 'weigend'
eri = mydf.get_eri((ki,kj,kk,kl))
print('ERI shape (%d,%d)' % eri.shape)

