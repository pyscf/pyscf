'''Fermi-Dirac smearing'''

import pyscf.pbc import gto, scf

cell = gto.Cell()
cell.atom = '''
He 0 0 1
He 1 0 1
'''
cell.basis = 'ccpvdz'
cell.a = numpy.eye(3) * 4
cell.gs = [8] * 3
cell.verbose = 4
cell.build()

nks = [2,1,1]
mf = pscf.KRHF(cell, cell.make_kpts(nks))
mf = scf.addons.smearing_(mf, sigma=.1, method='fermi')
mf.kernel()

