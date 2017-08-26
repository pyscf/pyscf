#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf

'''
Applying scalar relativistic effects by decorating the scf object with
scf.sfx2c function.

Similar to the density fitting decoration, the relativistic effects can also
be applied without affecting the existed scf object.
'''

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.sfx2c(scf.RHF(mol))
energy = mf.kernel()
print('E = %.12f, ref = -76.075429084850' % energy)


mol.spin = 1
mol.charge = 1
mol.build(0, 0)

mf = scf.UKS(mol).x2c()  # Using stream style
energy = mf.kernel()
print('E = %.12f, ref = -75.439160951099' % energy)


