#!/usr/bin/env python
#
# Author: Ardavan Farahvash <ardf.scf@gmail.com>
#
'''
Benggston-Neugebauer-Scheffler Dipole Corrections for surface calculations. 


Removes the leading-order error of polarized surfaces caused by the spurious
electrostatic interaction between periodic slab surface and its normal images. 

'''

import numpy as np
from pyscf.pbc import gto, dft
from pyscf.pbc.scf.addons import slab_dipole_correction

# Create a highly polar HF molecule unit cell
atom = "F 0.00 0.00 0.00; H 0.00 0.00 0.92"
cell = gto.Cell()
cell.atom = atom
cell.basis = "gth-szv"
cell.pseudo = "gth-pade"
cell.unit = "A"
cell.dimension = 3 
cell.verbose = 0

# Run large calculation  
print("--------- Running Huge Vacuum ---------")
cell.a = np.diag([10, 10, 200])
cell.build()

mf = dft.KUKS(cell, cell.make_kpts([1, 1, 1])).density_fit()
mf.xc = "pbe"
mf.kernel()
e0 = mf.kernel()
    
# uncorrected energy
print("--------- Running Uncorrected Calculations ---------")
for d in np.arange(5,18,3):
    cell.a = np.diag([10, 10, d])
    cell.build()

    mf = dft.KRKS(cell, cell.make_kpts([1, 1, 1])).density_fit()
    mf.xc = "pbe"
    mf.kernel()
    e = mf.kernel()-e0
    print(f"Uncorrected d={d}, E-E(d=200)={e:.6f} Ha")

print("--------- Running Corrected Calculations ---------")
# corrected energy
for d in np.arange(5,18,3):
    cell.a = np.diag([10, 10, d])
    cell.build()

    mf = dft.KRKS(cell, cell.make_kpts([1, 1, 1])).density_fit()
    mf = slab_dipole_correction(mf, dir_idx=2)
    mf.xc = "pbe"
    mf.kernel()
    e = mf.kernel()-e0
    print(f"Corrected d={d}, E-E(d=200)={e:.6f} Ha")