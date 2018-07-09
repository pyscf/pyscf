#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
The object returned by mf.nuc_grad_method() can be used to compute analytical
nuclear gradients.
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
grad = mf.nuc_grad_method()
grad.kernel()

mf = scf.UHF(mol).x2c()
mf.kernel()
grad = mf.nuc_grad_method()
grad.kernel()
