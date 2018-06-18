#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
The default XC functional library (libxc) supports the energy and nuclear
gradients for range separated functionals.  Nuclear Hessian and TDDFT gradients
need xcfun library.  See also example 32-xcfun_as_default.py for how to set
xcfun library as the default XC functional library.
'''

from pyscf import gto, dft

mol = gto.M(atom="H; F 1 1.", basis='631g')
mf = dft.UKS(mol)
mf.xc = 'CAMB3LYP'
mf.kernel()

mf.nuc_grad_method().kernel()


from pyscf.hessian import uks as uks_hess
# Switching to xcfun library on the fly
mf._numint.libxc = dft.xcfun
hess = uks_hess.Hessian(mf).kernel()
print(hess.reshape(2,3,2,3))


from pyscf import tdscf
# Switching to xcfun library on the fly
mf._numint.libxc = dft.xcfun
tdks = tdscf.TDA(mf)
tdks.nstates = 3
tdks.kernel()

tdks.nuc_grad_method().kernel()

