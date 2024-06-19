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

kmf = scf.KRHF(cell, kpts)
kmf.kernel()
#dm = kmf.init_guess_by_1e()
dm = kmf.make_rdm1()

dm_kpts = dm.copy()

print("dm = ", dm[0])
print("dm = ", dm[1])

#nuc = kmf.get_hcore(kpts=kpts)

vj, vk = kmf.with_df.get_jk(dm, kpts=kpts)

C = 25
prim_partition = [[0,1,2,3,4,5,6,7]]
#prim_partition = [[0,1]]
pbc_isdf_info  = isdf_linear_scaling_k.PBC_ISDF_Info_Quad_K(cell, kmesh=nk, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, rela_cutoff_QRCP=3e-3)
pbc_isdf_info.build_IP_local(c=C, m=5, group=prim_partition, Ls=[nk[0]*10, nk[1]*10, nk[2]*10])
pbc_isdf_info.build_auxiliary_Coulomb(debug=True)

vj2, vk2 = pbc_isdf_info.get_jk(dm, kpts = kpts)

print("vj  = ", vj[0, 0, :])
print("vj  = ", vj[1, 0, :])
print("vj2 = ", vj2[0, 0, :])
print("vj2 = ", vj2[1, 0, :])
print("vk  = ", vk[0, 0, :])
print("vk2 = ", vk2[0, 0, :])

diff  = numpy.linalg.norm(vj-vj2) / numpy.sqrt(vj.size)
diff2 = numpy.linalg.norm(vk-vk2) / numpy.sqrt(vk.size)

print("diff  = ", diff)
print("diff2 = ", diff2)

for i in range(vj.shape[0]):
    diff = numpy.linalg.norm(vj[i, :, :] - vj2[i, :, :]) / numpy.sqrt(vj[i].size)
    print(i, " diff  = ", diff)

for i in range(vk.shape[0]):
    diff = numpy.linalg.norm(vk[i, :, :] - vk2[i, :, :]) / numpy.sqrt(vk[i].size)
    print(i, " diff  = ", diff)

#exit(1)

#kmf.kernel()

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

#### test the accuracy of the ISDF 

atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        # ['C', (1.7834 , 1.7834 , 0.    )],
        # ['C', (2.6751 , 2.6751 , 0.8917)],
        # ['C', (1.7834 , 0.     , 1.7834)],
        # ['C', (2.6751 , 0.8917 , 2.6751)],
        # ['C', (0.     , 1.7834 , 1.7834)],
        # ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 
boxlen = 3.5668
prim_a = numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])

verbose = 10

from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell
from pyscf.pbc.df.isdf.isdf_linear_scaling_k import PBC_ISDF_Info_Quad_K

prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], basis=basis, pseudo=pseudo, ke_cutoff=KE_CUTOFF)
prim_mesh = prim_cell.mesh
mesh = [prim_mesh[0]*nk[0], prim_mesh[1]*nk[1], prim_mesh[2]*nk[2]]
cell = build_supercell(atm, prim_a, mesh=mesh, 
                                        Ls=nk,
                                        basis=basis, 
                                        pseudo=pseudo,
                                        ke_cutoff=KE_CUTOFF, verbose=verbose)


pbc_isdf_info = PBC_ISDF_Info_Quad_K(prim_cell, kmesh=nk, with_robust_fitting=False, aoR_cutoff=1e-8, direct=False, rela_cutoff_QRCP=3e-3)
pbc_isdf_info.build_IP_local(c=C, m=5, group=prim_partition, Ls=[nk[0]*10, nk[1]*10, nk[2]*10])
pbc_isdf_info.build_auxiliary_Coulomb(debug=True)

mf = scf.RHF(cell)
# mf = scf.addons.smearing_(mf, sigma=0.2, method='fermi')
mf.max_cycle = 16
mf.conv_tol = 1e-7

dm = mf.init_guess_by_1e()

from pyscf.pbc.df.isdf.isdf_linear_scaling_k_jk import _preprocess_dm

dm_real, _ = _preprocess_dm(pbc_isdf_info, dm)

print("dm_real = ", dm_real.shape)
print("dm      = ", dm.shape)
diff_dm = numpy.linalg.norm(dm_real - dm) / numpy.sqrt(dm.size)
print("diff_dm = ", diff_dm)


pbc_isdf_info.direct_scf = mf.direct_scf
mf.with_df = pbc_isdf_info

vj3, vk3 = mf.with_df.get_jk(dm, hermi=0, kpts=numpy.zeros((3)))

print("vj3 = ", vj3[0, :])
print("vk3 = ", vk3[0, :])