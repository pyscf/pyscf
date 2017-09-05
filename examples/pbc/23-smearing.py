#!/usr/bin/env python

'''Fermi-Dirac or Gaussian smearing for PBC SCF calculation'''

import numpy
from pyscf.pbc import gto, scf

cell = gto.Cell()
cell.atom = '''
He 0 0 1
He 1 0 1
'''
cell.basis = 'ccpvdz'
cell.a = numpy.eye(3) * 4
cell.verbose = 4
cell.build()

#
# Use scf.addons.smearing_ function to modify PBC (gamma-point or k-points)
# SCF object
#
nks = [2,1,1]
mf = scf.KRHF(cell, cell.make_kpts(nks))
mf = scf.addons.smearing_(mf, sigma=.1, method='fermi')
mf.kernel()
print('Entropy = %s' % mf.entropy)

#
# The smearing method and parameters can be modified at runtime
#
mf = scf.addons.smearing_(scf.UHF(cell))
mf.sigma = .1
mf.method = 'gauss'
mf.max_cycle = 2
mf.kernel()

mf.sigma = .05
mf.method = 'fermi'
mf.max_cycle = 50
mf.kernel()

