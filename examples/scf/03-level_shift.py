#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
For UHF method, level shift for alpha and beta spin can be different.
'''

from pyscf import gto, scf

mol = gto.M(
    atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
    basis = 'ccpvdz',
    verbose = 4
)

mf = scf.RHF(mol)
mf.level_shift = 0.2
mf.kernel()

#
# Same level shift is applied to alpha and beta
#
mf = scf.UHF(mol)
mf.level_shift = 0.2
mf.kernel()

#
# Break alpha and beta degeneracy by applying different level shift
#
mf = scf.UHF(mol)
mf.level_shift = (0.3, 0.2)
mf.kernel()
