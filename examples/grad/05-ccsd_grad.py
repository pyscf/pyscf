#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD analytical nuclear gradients.
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
postmf = cc.CCSD(mf).run()
grad = postmf.nuc_grad_method()
grad.kernel()

mf = scf.UHF(mol).x2c().run()
postmf = cc.CCSD(mf).run()
# PySCF-1.6.1 and newer supports the .Gradients method to create a grad
# object after grad module was imported. It is equivalent to call the
# .nuc_grad_method method.
from pyscf import grad
g = postmf.Gradients()
g.kernel()

# Perturbative triples require their own gradient class and lambda equations.
# Reuse the converged CCSD amplitudes and transformed integrals explicitly.
from pyscf.grad import ccsd_t, uccsd_t
from pyscf.cc import ccsd_t_lambda_slow, uccsd_t_lambda
rcc = cc.CCSD(scf.RHF(mol).run()).run()
eris = rcc.ao2mo()
_, l1, l2 = ccsd_t_lambda_slow.kernel(rcc, eris, rcc.t1, rcc.t2)
print('CCSD(T) gradient',
      ccsd_t.Gradients(rcc).kernel(rcc.t1, rcc.t2, l1, l2, eris=eris))
ucc = cc.UCCSD(rcc._scf.to_uhf()).run()
ueris = ucc.ao2mo()
_, ul1, ul2 = uccsd_t_lambda.kernel(ucc, ueris, ucc.t1, ucc.t2)
print('UCCSD(T) gradient',
      uccsd_t.Gradients(ucc).kernel(ucc.t1, ucc.t2, ul1, ul2, eris=ueris))
