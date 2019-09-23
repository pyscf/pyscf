#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CISD analytical nuclear gradients.
'''

from pyscf import gto, scf, ci

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
postmf = ci.CISD(mf).run()
grad = postmf.nuc_grad_method()
grad.kernel()

mf = scf.UHF(mol).x2c().run()
postmf = ci.CISD(mf).run()
grad = postmf.nuc_grad_method()
grad.kernel()
