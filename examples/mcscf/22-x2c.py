#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
Applying scalar relativistic effects for CASSCF by decorating the SCF or
CASSCF object with scf.sfx2c function.

See pyscf/examples/scf/21-x2c.py
'''

mol = gto.M(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.sfx2c(scf.RHF(mol))
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 8)
mc.kernel()
print('E = %.12f, ref = -76.129000779493' % mc.e_tot)

#
# Decorating CASSCF with scf.sfx2c has the same effects as decorating SCF object
#
mf = scf.RHF(mol)
mf.kernel()

mc = scf.sfx2c(mcscf.CASSCF(mf, 6, 8))
mc.kernel()
print('E = %.12f, ref = -76.129000779493' % mc.e_tot)

