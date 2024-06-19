#!/usr/bin/env python

'''
Mean field with k-points sampling

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.
'''

from pyscf.pbc import gto, scf, dft
import numpy

import pyscf.pbc.df.isdf.isdf_linear_scaling_k as isdf_linear_scaling_k

basis  = 'gth-szv'
pseudo = 'gth-pade'

KE_CUTOFF = 70

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    #atom = '''C     0.      0.      0.    
    #          C     0.8917  0.8917  0.8917''',
    basis  = basis,
    pseudo = pseudo,
    verbose = 10,
    ke_cutoff=KE_CUTOFF
)

nk = [1, 1, 3]  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
print("kpts = ", kpts)

kmf = scf.KUHF(cell, kpts)
kmf.kernel()
#dm = kmf.init_guess_by_1e()
dm = kmf.make_rdm1()

dm_kpts = dm.copy()

#print("dm = ", dm[0])
#print("dm = ", dm[1])

#nuc = kmf.get_hcore(kpts=kpts)

vj, vk = kmf.with_df.get_jk(dm, kpts=kpts)

C = 25
prim_partition = [[0,1,2,3,4,5,6,7]]
#prim_partition = [[0,1]]
pbc_isdf_info  = isdf_linear_scaling_k.PBC_ISDF_Info_Quad_K(cell, kmesh=nk, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, rela_cutoff_QRCP=3e-3)
pbc_isdf_info.build_IP_local(c=C, m=5, group=prim_partition, Ls=[nk[0]*10, nk[1]*10, nk[2]*10])
pbc_isdf_info.build_auxiliary_Coulomb(debug=True)

vj2, vk2 = pbc_isdf_info.get_jk(dm, kpts = kpts)

### perform SCF ###

print("kmf.kpts = ", kmf.kpts)
kmf.with_df = pbc_isdf_info
print("kmf.kpts = ", kmf.kpts)
kmf.max_cycle = 16
print("kmf.kpts = ", kmf.kpts)
kmf.conv_tol = 1e-7
print("kmf.kpts = ", kmf.kpts)
kmf.kernel()

exit(1)
