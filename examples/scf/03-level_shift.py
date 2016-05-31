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
O 0 0.     0
H 0 -2.757 2.587
H 0  2.757 2.587''',
    basis = 'ccpvdz',
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
