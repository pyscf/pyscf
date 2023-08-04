#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf

'''
Density fitting method by decorating the scf object with scf.density_fit function.

There is no flag to control the program to do density fitting for 2-electron
integration.  The way to call density fitting is to decorate the existed scf
object with scf.density_fit function.

NOTE scf.density_fit function generates a new object, which works exactly the
same way as the regular scf method.  The density fitting scf object is an
independent object to the regular scf object which is to be decorated.  By
doing so, density fitting can be applied anytime, anywhere in your script
without affecting the exsited scf object.

See also:
examples/df/00-with_df.py
examples/df/01-auxbasis.py 
'''

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.density_fit(scf.RHF(mol))
energy = mf.kernel()
print('E = %.12f, ref = -76.026744737355' % energy)

#
# Stream style: calling .density_fit method to return a DF-SCF object.
#
mf = scf.RHF(mol).density_fit()
energy = mf.kernel()
print('E = %.12f, ref = -76.026744737355' % energy)

#
# By default optimal auxiliary basis (if possible) or even-tempered gaussian
# functions are used fitting basis.  You can assign with_df.auxbasis to change
# the change the fitting basis.
#
mol.spin = 1
mol.charge = 1
mol.build(0, 0)
mf = scf.UKS(mol).density_fit()
mf.with_df.auxbasis = 'cc-pvdz-jkfit'
energy = mf.kernel()
print('E = %.12f, ref = -75.390366559552' % energy)

# Switch off density fitting
mf.with_df = False
energy = mf.kernel()
print('E = %.12f, ref = %.12f' % (energy, scf.UKS(mol).kernel()))
