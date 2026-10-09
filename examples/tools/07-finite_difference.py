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
# Hessian for a method that has an analytic gradient but no analytic second
# derivative. TDDFT is the common case: excited-state vibrational frequencies
# are only reachable this way.
#
mf = mol.RKS(xc='pbe0')
mf.grids.level = 5      # a Hessian needs a finer grid than the default
mf.grids.prune = None
mf.run(conv_tol=1e-13)
td = mf.TDA().run(nstates=5)
# Build the scanner to pin the root being differentiated
H = finite_diff.kernel(td.nuc_grad_method().as_scanner(state=1))
print('Finite difference Hessian of the first excited state:')
print(H)

#
# Finite difference Gradients as a PySCF builtin Gradients object
#
mf = mol.RHF()
finite_diff.Gradients(mf).run()
#
# This object can be used in the geometry optimization processes
#
finite_diff.Gradients(mf).optimizer().run()
