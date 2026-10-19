#!/usr/bin/env python

'''
A simple example to run EPH calculation using finite difference.
'''
import numpy as np
from pyscf import gto, dft, eph
from pyscf.eph.eph_fd import kernel

mol = gto.M(atom='N 0 0 0; N 0 0 2.100825', basis='def2-svp', verbose=4, unit="bohr")
# this is a pre-computed relaxed molecule
# for geometry relaxation, refer to pyscf/example/geomopt
mf = dft.RKS(mol, xc='pbe,pbe')
mf.run()

grad = mf.nuc_grad_method().kernel()
assert (abs(grad).sum()<1e-5) # making sure the geometry is relaxed

mat, omega = kernel(mf, disp=1e-4, mo_rep=True)
print(mat.shape, omega)

# Compare finite-difference and analytic couplings in both representations.
# Vibrational eigenvectors can differ by a sign, so compare each mode after
# choosing the better of the two signs.
fd_ao, _ = kernel(mf, disp=1e-4)
analytic_ao, _ = eph.EPH(mf).kernel()
analytic_mo, _ = eph.EPH(mf).kernel(mo_rep=True)
for label, fd, analytic in (('AO', fd_ao, analytic_ao),
                            ('MO', mat, analytic_mo)):
    errors = [min(np.linalg.norm(a - b), np.linalg.norm(a + b))
              for a, b in zip(fd, analytic)]
    print(label, 'finite-difference errors by mode:', errors)
