#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
Applying scalar relativistic effects for CASSCF by decorating the SCF or
CASSCF object with .x2c method.

See pyscf/examples/scf/21-x2c.py
'''

mol = gto.M(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.RHF(mol).x2c()
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 8)
mc.kernel()
print('E = %.12f, ref = -76.128478294795' % mc.e_tot)

#
# Decorating CASSCF with .x2c method has the same effects as decorating SCF object
#
mf = scf.RHF(mol)
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 8).x2c()
mc.kernel()
print('E = %.12f, ref = -76.128478294795' % mc.e_tot)

