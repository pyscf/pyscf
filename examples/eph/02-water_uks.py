#!/usr/bin/env python

"""Compare UKS and RKS electron-phonon couplings for closed-shell water.

Use the precomputed relaxed B3LYP/STO-3G geometry from pyscf.eph.uks.
Compare alpha/beta couplings and restricted/unrestricted results in the AO basis.
"""

import numpy as np
from pyscf import gto, dft
from pyscf.eph.uks import EPH
from pyscf.eph.rks import EPH as REPH
mol = gto.M()
mol.atom = [['O', [0.000000000000, 0.000000002577,0.868557119905]],
            ['H', [0.000000000000,-1.456050381698,2.152719488376]],
            ['H', [0.000000000000, 1.456050379121,2.152719486067]]]

mol.unit = 'Bohr'
mol.basis = 'sto3g'
mol.verbose=4
mol.build() # this is a pre-computed relaxed geometry

mf = dft.UKS(mol)
mf.grids.level=6
mf.xc = 'b3lyp'
mf.conv_tol = 1e-16
mf.conv_tol_grad = 1e-10
mf.kernel()

grad = mf.nuc_grad_method().kernel()
print("Nuclear energy gradient (Hartree/Bohr):")
print(grad)


myeph = EPH(mf)
(epha, ephb), omega_uks = myeph.kernel()

mf = dft.RKS(mol)
mf.grids.level=6
mf.xc = 'b3lyp'
mf.conv_tol = 1e-16
mf.conv_tol_grad = 1e-10
mf.kernel()

myeph = REPH(mf)
rmat, omega_rks = myeph.kernel()
print('UKS/RKS frequency difference:', np.linalg.norm(omega_uks-omega_rks))
print('UKS alpha/beta coupling difference:', np.linalg.norm(epha-ephb))
# Normal-mode eigenvectors can differ by a sign between calculations.
# This water geometry has nondegenerate vibrational modes.
for i in range(len(rmat)):
    print('Mode %d UKS/RKS coupling difference:' % i,
          min(np.linalg.norm(epha[i]-rmat[i]), np.linalg.norm(epha[i]+rmat[i])))
