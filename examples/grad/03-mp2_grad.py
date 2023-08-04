#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
MP2 analytical nuclear gradients.
'''

from pyscf import gto, scf, mp

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
postmf = mp.MP2(mf).run()
g = postmf.nuc_grad_method()
g.kernel()

mf = scf.UHF(mol).x2c().run()
postmf = mp.MP2(mf).run()
# PySCF-1.6.1 and newer supports the .Gradients method to create a grad
# object after grad module was imported. It is equivalent to call the
# .nuc_grad_method method.
from pyscf import grad
g = postmf.Gradients()
g.kernel()
