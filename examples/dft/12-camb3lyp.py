#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density functional calculations can be run with either the default
backend library, libxc, or an alternative library, xcfun. See also
example 32-xcfun_as_default.py for how to set xcfun as the default XC
functional library.

'''

from pyscf import gto, dft
from pyscf.hessian import uks as uks_hess
from pyscf import tdscf

mol = gto.M(atom="H; F 1 1.", basis='631g')

# Calculation using libxc
mf = dft.UKS(mol)
mf.xc = 'CAMB3LYP'
mf.kernel()
mf.nuc_grad_method().kernel()

# We can also evaluate the geometric hessian
hess = uks_hess.Hessian(mf).kernel()
print(hess.reshape(2,3,2,3))

# or TDDFT gradients
tdks = tdscf.TDA(mf)
tdks.nstates = 3
tdks.kernel()
tdks.nuc_grad_method().kernel()

# Switch to the xcfun library on the fly
mf._numint.libxc = dft.xcfun
# Repeat the geometric hessian
hess = uks_hess.Hessian(mf).kernel()
print(hess.reshape(2,3,2,3))
# and the TDDFT gradient calculation
tdks = tdscf.TDA(mf)
tdks.nstates = 3
tdks.kernel()
tdks.nuc_grad_method().kernel()
