#!/usr/bin/env python
#
# Author: Huanchen Zhai <hczhai.ok@gmail.com>
#

'''
Examples of internally contracted CASPT2 calculations.

This script demonstrates:
    - CASPT2 with and without IPEA shift
    - CASPT2 with outcore (ccvv part only) for reducing memory usage
    - CASPT2 with and without density fitting
    - CASPT2 for each state from a state-averaged CASSCF
'''

# C10H12 reference:
#   Chem. Sci. (2019) 10 (6): 1716–1723.
#   https://doi.org/10.1039/c8sc03569e
#   Geometry from Supplementary Material Table S6 (n = 5)
#   Reference energy from Supplementary Material Table S4 (n = 5)
#   E(CASSCF) = -385.753473, E(CASPT2) = -386.956445

from pyscf import gto, scf, mcscf, mrpt, mp, fci
import numpy as np

geom = """
C -5.544122  0.214299 0.000000
C  5.544122 -0.214299 0.000000
C -4.350171 -0.402736 0.000000
C  4.350171  0.402736 0.000000
H -5.622764  1.298325 0.000000
H  5.622764 -1.298325 0.000000
H -6.474236 -0.343751 0.000000
H  6.474236  0.343751 0.000000
H -4.316850 -1.492261 0.000000
H  4.316850  1.492261 0.000000
C -3.074155  0.278469 0.000000
C  3.074155 -0.278469 0.000000
C -1.872943 -0.352359 0.000000
C  1.872943  0.352359 0.000000
H -3.097453  1.368452 0.000000
H  3.097453 -1.368452 0.000000
H -1.855664 -1.442745 0.000000
H  1.855664  1.442745 0.000000
C -0.600949  0.317418 0.000000
C  0.600949 -0.317418 0.000000
H -0.616498  1.407635 0.000000
H  0.616498 -1.407635 0.000000
"""

mol = gto.M(atom=geom, basis='6-31+g(d,p)', symmetry=True, verbose=3, max_memory=10000)
mf = scf.RHF(mol).run(conv_tol=1e-12)

mc = mcscf.CASSCF(mf, 10, 10)
mc.fcisolver.conv_tol = 1e-14
mc.conv_tol = 1e-12
mc.verbose = 4

mx = mp.MP2(mf, frozen=10).run()
# make symmetry-adapted natural orbtials
mo = mf.mo_coeff @ mrpt.caspt2.disjoint_eigh(mx.make_rdm1())[1][:, ::-1]

mc.kernel(mo)
print('=== E(CASSCF) = %15.8f  REF = %15.8f' % (mc.e_tot, -385.75332172))

# CASPT2 with IPEA shift = 0.25 (incore)
mr = mrpt.caspt2.CASPT2(mc, frozen=10, ipea_shift=0.25)
mr.max_cycle = 20
mr.kernel()
print('=== E(CASPT2-IPEA) = %15.8f  REF = %15.8f' % (mr.e_tot, -386.95650687))

# CASPT2 with IPEA shift = 0.25 (outcore)
mc.max_memory = 4000
mr = mrpt.caspt2.CASPT2(mc, frozen=10, ipea_shift=0.25)
mr.max_cycle = 20
mr.kernel()
print('=== E(CASPT2-IPEA) = %15.8f  REF = %15.8f' % (mr.e_tot, -386.95650687))

# CASPT2 without IPEA shift
mr = mrpt.caspt2.CASPT2(mc, frozen=10, ipea_shift=0.0)
mr.max_cycle = 20
mr.kernel()
print('=== E(CASPT2) = %15.8f  REF = %15.8f' % (mr.e_tot, -386.96298177))

# CASPT2 with density fitting
mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
mc = mcscf.CASSCF(mf, 10, 10).density_fit()
mc.fcisolver.conv_tol = 1e-14
mc.conv_tol = 1e-12
mc.verbose = 4
mx = mp.MP2(mf, frozen=10).run()
# make symmetry-adapted natural orbtials
mo = mf.mo_coeff @ mrpt.caspt2.disjoint_eigh(mx.make_rdm1())[1][:, ::-1]
mc.kernel(mo)
print('=== E(DF-CASSCF) = %15.8f  REF = %15.8f' % (mc.e_tot, -385.75280633))

mr = mrpt.caspt2.CASPT2(mc, frozen=10, ipea_shift=0.0)
mr.max_cycle = 20
mr.kernel()
print('=== E(DF-CASPT2) = %15.8f  REF = %15.8f' % (mr.e_tot, -386.96185175))

# CASPT2 for each state from a state-averaged CASSCF
mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, symmetry='d2h', verbose=3)
mf = scf.RHF(mol).run(conv_tol=1E-12)

mc = mcscf.CASSCF(mf, 6, 8)
mc.fcisolver = fci.addons.fix_spin(mc.fcisolver, shift=0.2, ss=2)
mc.fcisolver.conv_tol = 1e-14
mc.conv_tol = 1e-12
mc.fcisolver.nroots = 3
mc = mcscf.state_average_(mc, [1.0 / 3] * 3)
mc.kernel()
print('=== E(SA-CASSCF) = %15.8f  REF = %15.8f' % (mc.e_tot, -149.21530829))

ref_eners = [-149.98084840, -149.41476256, -149.15130483]
for iroot in range(mc.fcisolver.nroots):
    mr = mrpt.caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0, root=iroot)
    mr.verbose = 4
    mr.conv_tol_normt = 1e-12
    mr.max_cycle = 20
    mr.kernel()
    print('=== ROOT %d E(CASPT2) = %15.8f  REF = %15.8f' % (iroot, mr.e_tot, ref_eners[iroot]))
