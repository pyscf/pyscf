#!/usr/bin/env python

'''
MP2 with k-points sampling when Brillouin Zone symmetry is considered
'''

import numpy
from pyscf.pbc import gto, scf, mp

cell = gto.M(
    a = numpy.asarray([[0.0, 2.6935121974, 2.6935121974],
                       [2.6935121974, 0.0, 2.6935121974],
                       [2.6935121974, 2.6935121974, 0.0]]),
    atom = '''Si  0.0000000000 0.0000000000 0.0000000000
              Si  1.3467560987 1.3467560987 1.3467560987''',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    mesh = [24,24,24],
    verbose = 5,
    space_group_symmetry = True,
)

nk = [2,2,2]
kpts = cell.make_kpts(nk,
                      space_group_symmetry=True,
                      time_reversal_symmetry=True)

kmf = scf.KRHF(cell, kpts)
kmf.kernel()

kmp2 = mp.KMP2(kmf)
kmp2.kernel()
print("KMP2 energy (per unit cell) =", kmp2.e_tot)
