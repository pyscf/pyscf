#!/usr/bin/env python

'''
Finite difference driver for generating gradients and Hessians for PySCF methods.

If the input method is used for energy computation, the finite_diff.kernel 
can compute the gradients using finite difference techniques. Conversely, if 
the input method is designed for nuclear gradients, the finite_diff.kernel 
can compute the Hessian using finite difference methods.
'''

import pyscf
from pyscf.tools import finite_diff
mol = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz')

#
# Gradients
#
method = mol.RHF().CCSD()
de = finite_diff.kernel(method)
print('Finite difference Gradients:')
print(de)

print('Analytical Gradients:')
print(mol.CCSD().Gradients().kernel())

#
# Hessian
#
method = mol.RHF().Gradients()
H = finite_diff.kernel(method)
print('Finite difference Hessian:')
print(H)

print('Analytical Hessian:')
print(mol.RHF().run().Hessian().kernel())

#
# Finite difference Gradients as a PySCF builtin Gradients object
#
mf = mol.RHF()
Gradients(mf).run()
#
# This object can be used in the geometry optimization processes
#
Gradients(mf).optimizer().run()
