#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto
from pyscf import scf

'''
Mixing decoration, for density fitting, scalar relativistic effects, and
second order (Newton-Raphson) SCF.

Density fitting and scalar relativistic effects can be applied together,
regardless to the order you apply the decoration.

NOTE the second order SCF (New in version 1.1) decorating operation are not
commutable with scf.density_fit operation
        [scf.density_fit, scf.sfx2c1e    ] == 0
        [scf.newton     , scf.sfx2c1e    ] != 0
        [scf.newton     , scf.density_fit] != 0
* scf.density_fit(scf.newton(scf.RHF(mol))) is the SOSCF for regular 2e
  integrals, but with density fitting integrals for the Hessian.  It's an
  approximate SOSCF optimization method;
* scf.newton(scf.density_fit(scf.RHF(mol))) is the exact second order
  optimization for the given scf object which is a density-fitted-scf method.
  The SOSCF is not an approximate scheme.
* scf.density_fit(scf.newton(scf.density_fit(scf.RHF(mol))), auxbasis='ahlrichs')
  is an approximate SOSCF scheme for the given density-fitted-scf method.
  Here we use small density fitting basis (ahlrichs cfit basis) to approximate
  the Hessian for the large-basis-density-fitted-scf scheme.
'''

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

#
# 1. spin-free X2C-HF with density fitting approximation on 2E integrals
#
mf = scf.density_fit(scf.sfx2c1e(scf.RHF(mol)))
mf = scf.RHF(mol).x2c().density_fit()  # Stream style
energy = mf.kernel()
print('E = %.12f, ref = -76.075408156180' % energy)

#
# 2. spin-free X2C correction for density-fitting HF.  Since X2C correction is
# commutable with density fitting operation, it is fully equivalent to case 1.
#
mf = scf.sfx2c1e(scf.density_fit(scf.RHF(mol)))
mf = scf.RHF(mol).density_fit().x2c()  # Stream style
energy = mf.kernel()
print('E = %.12f, ref = -76.075408156180' % energy)

#
# 3. The order to apply X2C or newton method matters.  If relativistic effects
# need to be included in the calculation, you should first call x2c then
# newton method.
#
e1 = scf.RHF(mol).kernel()
e2 = scf.RHF(mol).x2c().kernel()
print('E(NR)         = %.12f  E(X2C)        = %.12f' % (e1, e2))
e1 = scf.RHF(mol).newton().x2c().kernel()
e2 = scf.RHF(mol).x2c().newton().kernel()
print('E(Newton,X2C) = %.12f  E(X2C,Newton) = %.12f' % (e1, e2))

#
# 4. Newton method for non-relativistic HF
#
mf = scf.newton(scf.RHF(mol))
mf = scf.RHF(mol).newton()  # Stream style
energy = mf.kernel()
print('E = %.12f, ref = -76.026765673120' % energy)

#
# 5. Newton method for non-relativistic HF with density fitting for orbital
# hessian of newton solver.  Note the answer is equal to case 3, but the
# solver "mf" is different.
#
mf = scf.density_fit(scf.newton(scf.RHF(mol)))
mf = scf.RHF(mol).newton().density_fit()
energy = mf.kernel()
print('E = %.12f, ref = -76.026765673120' % energy)

#
# 6. Newton method to solve the density-fitting approximated HF object.  There
# is no approximation for newton method (orbital hessian).  Note the density
# fitting is applied on HF object only.  It does not affect the Newton solver.
#
mf = scf.newton(scf.density_fit(scf.RHF(mol)))
mf = scf.RHF(mol).density_fit().newton()
energy = mf.kernel()
print('E = %.12f, ref = -76.026744737357' % energy)

#
# 7. Newton method for density-fitting HF, and the hessian of Newton solver is
# also approximated with density fitting.  Note the anwser is equivalent to
# case 6, but the solver "mf" is different.  Here the fitting basis for HF and
# Newton solver are different.  HF is approximated with the default density
# fitting basis (Weigend cfit basis).  Newton solver is approximated with
# Ahlrichs cfit basis.
#
mf = scf.density_fit(scf.newton(scf.density_fit(scf.RHF(mol))), 'ahlrichs')
mf = scf.RHF(mol).density_fit().newton().density_fit(auxbasis='ahlrichs')
energy = mf.kernel()
print('E = %.12f, ref = -76.026744737357' % energy)

