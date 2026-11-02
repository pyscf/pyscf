#!/usr/bin/env python

'''
A simple example to run EPH calculation.
'''

import numpy as np
from pyscf import gto, dft, eph

mol = gto.M()
mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
            ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
            ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
mol.unit = 'Bohr'
mol.basis = 'sto3g'
mol.verbose = 4
mol.build() # this is a pre-computed relaxed geometry

mf = mol.RHF()
mf.conv_tol = 1e-16
mf.conv_tol_grad = 1e-10
mf.kernel()

myeph = eph.EPH(mf)
grad = mf.nuc_grad_method().kernel()
print("Force on the atoms/au:")
print(grad)
ephmat, omega = myeph.kernel(mo_rep=True)
print(np.amax(ephmat))


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

# Closed-shell unrestricted EPH gives alpha and beta coupling matrices.
water = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
              basis='sto3g')
uhf = water.UHF().run()
(eph_a, eph_b), omega_uhf = eph.EPH(uhf).kernel()
print('UHF alpha/beta difference:', np.linalg.norm(eph_a - eph_b))
rhf_ao, _ = eph.EPH(water.RHF().run()).kernel()
print('UHF/RHF AO difference by mode:',
      [min(np.linalg.norm(a - r), np.linalg.norm(a + r))
       for a, r in zip(eph_a, rhf_ao)])

# The same spin-resolved interface works with unrestricted KS.
uks = dft.UKS(mol, xc='pbe,pbe').run()
(eph_a, eph_b), omega_uks = eph.EPH(uks).kernel()
print('UKS alpha/beta difference:', np.linalg.norm(eph_a - eph_b))
print('UKS/RKS AO difference by mode:',
      [min(np.linalg.norm(a - r), np.linalg.norm(a + r))
       for a, r in zip(eph_a, mat)])
