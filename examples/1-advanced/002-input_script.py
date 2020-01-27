#!/usr/bin/env python

'''
This example collects tricks that can be used in the pyscf input script.
'''

import pyscf


#
# 1. Import all pyscf modules
#

# This statement will import all methods into Mole class. These methods can be
# accessed as the attributes/methods of Mole object.
from pyscf import __all__
mol = pyscf.gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-311g')
print(mol.HF())
print(mol.KS().ddCOSMO())
print(mol.TDHF())
print(mol.MP2(frozen=2))


#
# 2. Stream operations.  There are three stream functions ".set", ".run",
# ".apply" to pipeline the computing streams.
# * Method .set assigns the attributes of the object using its keyword
#   arguments.
# * Method .run calls the kernel function of the object. Arguments of .run are
#   passed to kernel function if they were specified. Keyword arguments of .run
#   are passed to .set method if specified.
# * Method .apply makes up a function call with its argument, e.g. a.apply(f)
#   is transformed to  f(a)
#
# Using stream operations, we can write multiple computing tasks in one line.
# E.g. the following statements can be compressed into a stream
# >>> from pyscf import scf, mp2, grad
# >>> mf = scf.RHF(mol)
# >>> mf.conv_tol=1e-7
# >>> mf.kernel()
# >>> mp = mp2.MP2(mf, frozen=2)
# >>> mp.max_memory = 100
# >>> mp.kernel()
# >>> mp_grad = grad.mp2.Gradients(mp)
# >>> mp_grad.kernel()

mol.RHF().run(conv_tol=1e-7).MP2(frozen=2).run(max_memory=100).Gradients().run()

# Another example of pipelined stream
# >>> from pyscf import scf, dft, tdscf
# >>> mf = dft.KS(mol).density_fit()
# >>> # This function removes possible basis linear dependency
# >>> mf = scf.addons.remove_linear_dep_(mf)
# >>> mf.conv_tol=1e-6
# >>> mf.xc = 'blyp'
# >>> mf.kernel()
# >>> td = tdscf.TDA(mf)
# >>> td.nstates = 5
# >>> mp.kernel()
mol.KS() \
    .set(conv_tol=1e-6, xc='blyp') \
    .density_fit() \
    .apply(pyscf.scf.addons.remove_linear_dep_) \
    .run() \
    .TDA() \
    .run(nstates=5)

