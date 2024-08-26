#!/usr/bin/env python

'''
G0W0 with k-points sampling
'''

from functools import reduce
import numpy
from pyscf.pbc import gto, scf, gw

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

#
# KDFT and KGW with 2x2x2 k-points
#
kpts = cell.make_kpts([2,2,2])
kmf = scf.KRKS(cell).density_fit()
kmf.kpts = kpts
emf = kmf.kernel()

# Default is AC frequency integration
mygw = gw.KRGW(kmf)
mygw.kernel()
print("KRGW energies =", mygw.mo_energy)

# With CD frequency integration
#mygw = gw.KRGW(kmf, freq_int='cd')
#mygw.kernel()
#print("KRGW-CD energies =", mygw.mo_energy)

