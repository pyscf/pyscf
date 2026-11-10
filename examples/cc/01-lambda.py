#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD and CCSD(T) lambda equation
'''

import pyscf

mol = pyscf.M(
    atom = '''
O    0.   0.       0.
H    0.   -0.757   0.587
H    0.   0.757    0.587''',
    basis = 'cc-pvdz')
mf = mol.RHF().run()
cc = mf.CCSD().run()
#
# Solutions for CCSD Lambda equations are saved in cc.l1 and cc.l2
#
cc.solve_lambda()
print(cc.l1.shape)
print(cc.l2.shape)

###
#
# Compute CCSD(T) lambda with ccsd_t-slow implementation
# (as of pyscf v1.7)
#
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
conv, l1, l2 = ccsd_t_lambda.kernel(cc, cc.ao2mo(), cc.t1, cc.t2, tol=1e-8)

# UCCSD and GCCSD solve their lambda equations through the same interface.
# The spin-resolved UCCSD amplitudes are tuples; GCCSD uses spin orbitals.
ucc = mf.to_uhf().CCSD().run()
ucc.solve_lambda()
gcc = mf.to_ghf().CCSD().run()
gcc.solve_lambda()
print('UCCSD lambda blocks', len(ucc.l1), len(ucc.l2))
print('GCCSD lambda shapes', gcc.l1.shape, gcc.l2.shape)

# The same collinear RHF reference was converted above, so the lambda
# amplitudes can be compared after unpacking the generalized spin orbitals.
# orbspin records their actual ordering; do not assume alternating spins.
from pyscf.cc import addons, uccsd_t_lambda
orbspin = gcc.mo_coeff.orbspin
l1a, l1b = addons.spin2spatial(gcc.l1, orbspin)
l2aa, l2ab, l2bb = addons.spin2spatial(gcc.l2, orbspin)
print('UCCSD/GCCSD alpha lambda difference', abs(l1a - ucc.l1[0]).max())
print('UCCSD/GCCSD mixed-spin lambda difference', abs(l2ab - ucc.l2[1]).max())
gl1 = addons.spatial2spin(ucc.l1, orbspin)
print('Converted UCCSD/GCCSD lambda difference', abs(gl1 - gcc.l1).max())

# Supply converged amplitudes and reuse MO integrals explicitly.
ueris = ucc.ao2mo()
conv, ul1, ul2 = uccsd_t_lambda.kernel(ucc, ueris, ucc.t1, ucc.t2)
print('UCCSD(T) lambda converged', conv)
