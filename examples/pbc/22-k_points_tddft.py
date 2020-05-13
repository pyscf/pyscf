#!/usr/bin/env python

'''
TDDFT with k-point sampling or at an individual k-point

(This feature is in testing. We observe numerical stability problem in TDDFT
diagonalization.)

'''

from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import tdscf

cell = gto.Cell()
cell.unit = 'B'
cell.atom = '''
C  0.          0.          0.
C  1.68506879  1.68506879  1.68506879
'''
cell.a = '''
0.          3.37013758  3.37013758
3.37013758  0.          3.37013758
3.37013758  3.37013758  0.
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.build()
mf = scf.KRHF(cell, cell.make_kpts([2,2,2]))
mf.run()

td = tdscf.KTDA(mf)
td.nstates = 5
td.verbose = 5
print(td.kernel()[0] * 27.2114)

td = tdscf.KTDDFT(mf)
td.nstates = 5
td.verbose = 5
print(td.kernel()[0] * 27.2114)

mf = scf.RHF(cell)
mf.kernel()
td = tdscf.TDA(mf)
td.kernel()

# TODO:
#kpt = cell.get_abs_kpts([0.25, 0.25, 0.25])
#mf = scf.RHF(cell, kpt=kpt)
#mf.kernel()
#td = tdscf.TDA(mf)
#td.kernel()
