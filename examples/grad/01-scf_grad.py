#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
SCF analytical nuclear gradients calculation. SCF gradients object can be
created by calling .nuc_grad_method() or .Gradients() for pyscf-1.6.1 and
newer.
'''

from pyscf import gto, scf

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = '631g')

mf = scf.RHF(mol)
mf.kernel()
g = mf.nuc_grad_method()
g.kernel()

mf = scf.UHF(mol).x2c()
mf.kernel()
# PySCF-1.6.1 and newer supports the .Gradients method to create a grad
# object after grad module was imported. It is equivalent to call the
# .nuc_grad_method method.
from pyscf import grad
g = mf.Gradients()
g.kernel()
