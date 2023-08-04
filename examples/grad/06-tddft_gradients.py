#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
TDDFT analytical nuclear gradients.
'''

from pyscf import gto, scf, dft, tddft

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
postmf = tddft.TDHF(mf).run()
g = postmf.nuc_grad_method()
g.kernel(state=1)

mf = dft.RKS(mol).x2c().set(xc='pbe0').run()
# Switch to xcfun because 3rd order GGA functional derivative is not
# available in libxc
mf._numint.libxc = dft.xcfun
postmf = tddft.TDDFT(mf).run()
# PySCF-1.6.1 and newer supports the .Gradients method to create a grad
# object after grad module was imported. It is equivalent to call the
# .nuc_grad_method method.
from pyscf import grad
g = postmf.Gradients()
g.kernel(state=1)

#mf = scf.UHF(mol).x2c().run()
#postmf = tddft.TDHF(mf).run()
#g = postmf.nuc_grad_method()
#g.kernel()
