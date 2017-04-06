#!/usr/bin/env python

'''
CCSD with k-point sampling
'''

from pyscf.pbc import gto, scf, cc

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
cell.gs = [7]*3
cell.build()

#
# Running HF and CCSD for single k-point
#
kpts = cell.get_abs_kpts([0.25, 0.25, 0.25])
kmf = scf.KRHF(cell, exxdiv=None)
kmf.kpts = kpts
ehf = kmf.kernel()

mycc = cc.KRCCSD(kmf)
mycc.kernel()
print("KRCCSD energy (per unit cell) =", mycc.e_tot)

#
# Running HF and CCSD with 2x2x2 k-points
#
kpts = cell.make_kpts([2,2,2])
kmf = scf.KRHF(cell, exxdiv=None)
kmf.kpts = kpts
ehf = kmf.kernel()

mycc = cc.KCCSD(kmf)
mycc.kernel()
print("KRCCSD energy (per unit cell) =", mycc.e_tot)

