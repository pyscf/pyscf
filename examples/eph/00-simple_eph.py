#!/usr/bin/env python

'''
A simple example to run EPH calculation.
'''
from pyscf import gto, dft, eph
mol = gto.M(atom='N 0 0 0; N 0 0 2.100825', basis='def2-svp', verbose=4, unit="bohr")
# this is a pre-computed relaxed molecule
# for geometry relaxation, refer to pyscf/example/geomopt
mf = dft.RKS(mol, xc='pbe,pbe')
mf.run()

grad = mf.nuc_grad_method().kernel()
assert (abs(grad).sum()<1e-5) # making sure the geometry is relaxed

myeph = eph.EPH(mf)
mat, omega = myeph.kernel()
print(mat.shape, omega)
