#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Applying scalar relativistic effects by decorating the scf object with
.x2c method.

Similar to the density fitting decoration, the relativistic effects can also
be applied without affecting the existed scf object.
'''

from pyscf import gto
from pyscf import scf

mol = gto.M(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.RHF(mol).x2c().run()
print('E = %.12f, ref = -76.075429084850' % mf.e_tot)


mol.spin = 1
mol.charge = 1
mol.build(0, 0)

# .x2c1e is the same to the .x2c method in the current implementation
mf = scf.UKS(mol).x2c1e()
energy = mf.kernel()
print('E = %.12f, ref = -75.439160951099' % energy)

# Switch off x2c
mf.with_x2c = False
energy = mf.kernel()
print('E = %.12f, ref = %.12f' % (energy, scf.UKS(mol).kernel()))
