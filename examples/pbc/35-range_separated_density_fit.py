#!/usr/bin/env python

'''
The basic usage of range-separated Gaussian density fitting (RSGDF or simply
RSDF), including choosing an auxiliary basis, initializing 3c integrals, saving
and reading 3c integrals, jk build, ao2mo, etc is the same as that of the
GDF module. Please refer to the following examples for details:

- 21-k_points_all_electron_scf.py   # SCF
- 30-ao_integrals.py                # compute ao integrals
- 35-gaussian_density_fit.py        # auxiliary basis, save & load cderi,
                                    # and loop over 3c integrals

This script highlights special settings of the RSGDF module.

Note: currently RSDF does NOT support low-dimensional systems (0D ~ 2D).
'''

import numpy as np
from pyscf import gto as mol_gto
from pyscf.pbc import gto, scf, cc, df, mp

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'cc-pvdz'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 6
cell.build()

kmesh = [2,1,1]
kpts = cell.make_kpts(kmesh)

#
# Use the 'rs_density_fit' method provided by all PBC SCF classes (including
# DFT) for using RSDF to handle the density fitting.
#
mf = scf.KRHF(cell, kpts).rs_density_fit()
mf.kernel()


#
# One can also initialize a RSDF instance separately and overwrite 'SCF.with_df'
#
mydf = df.RSDF(cell, kpts)
mf = scf.KRKS(cell, kpts).rs_density_fit()
mf.with_df = mydf
mf.xc = "pbe"
mf.kernel()


#
# RSDF calculates the DF 3c integrals in two parts:
#     j3c = j3c_SR + j3c_LR
# where SR and LR stand for short-range and long-range, respectively.
# The parameter 'omega' determines the cost of the two parts: the larger omega
# is, the faster (slower) the SR (LR) part is computed, and vice versa.
# By default, the code will determine an optimal value for 'omega' that is
# suitable for most cases. The user can nonetheless tests the effect of using
# different values of 'omega'. The few lines below do this and verify that the
# results (in terms of both the HF and the MP2 energies) are not affected.
# In the output file, you can also
#     grep -a "CPU time for j3c" [output_file]
# to see how the DF initialization time is affected by using different omega.
#
omegas = np.array([0.3, 0.5, 0.7, 0.9])
escfs = np.zeros_like(omegas)
emp2s = np.zeros_like(omegas)
for i,omega in enumerate(omegas):
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.omega = omega
    mf.kernel()
    escfs[i] = mf.e_tot
    mmp = mp.KMP2(mf)
    mmp.with_t2 = False
    mmp.kernel()
    emp2s[i] = mmp.e_corr
for omega, escf, emp2 in zip(omegas, escfs, emp2s):
    print("%.2f  %.10f  %.10f" % (omega, escf, emp2))
maxdiffescf = escfs.max()-escfs.min()
maxdiffemp2 = emp2s.max()-emp2s.min()
print("Maximum difference in SCF energy: %.10f" % (maxdiffescf))
print("Maximum difference in MP2 energy: %.10f" % (maxdiffemp2))
''' example output:
0.30  -75.3226526450  -0.2242441141
0.50  -75.3226526440  -0.2242441145
0.70  -75.3226526451  -0.2242441148
0.90  -75.3226526455  -0.2242441143
Maximum difference in SCF energy: 0.0000000015
Maximum difference in MP2 energy: 0.0000000007
'''
