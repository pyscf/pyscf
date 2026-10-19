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

# Periodic finite differences use a k-point SCF object, even at Gamma.
# This illustrative diamond geometry should be relaxed before interpreting
# its mode frequencies quantitatively. Displacements are in atomic units.
from pyscf.pbc import gto as pbcgto, dft as pbcdft
from pyscf.pbc.eph import eph_fd as pbc_eph_fd
cell = pbcgto.M(
    atom='C 0 0 0; C 1.685068664391 1.685068664391 1.685068664391',
    a=[[0, 3.370137329, 3.370137329],
       [3.370137329, 0, 3.370137329],
       [3.370137329, 3.370137329, 0]],
    unit='Bohr', basis='gth-szv', pseudo='gth-pade')
kmf = pbcdft.KRKS(cell, cell.make_kpts([1, 1, 1]))
kmf.conv_tol = 1e-12
kmf.kernel()
periodic_mat, periodic_omega = pbc_eph_fd.kernel(kmf, mo_rep=True)
print('Periodic MO couplings and mode frequencies', periodic_mat.shape, periodic_omega)
