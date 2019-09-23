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
grad = postmf.nuc_grad_method()
grad.kernel(state=1)

mf = dft.RKS(mol).x2c().set(xc='pbe0').run()
# Switch to xcfun because 3rd order GGA functional derivative is not
# available in libxc
mf._numint.libxc = dft.xcfun
postmf = tddft.TDDFT(mf).run()
grad = postmf.nuc_grad_method()
grad.kernel(state=1)

#mf = scf.UHF(mol).x2c().run()
#postmf = tddft.TDHF(mf).run()
#grad = postmf.nuc_grad_method()
#grad.kernel()
