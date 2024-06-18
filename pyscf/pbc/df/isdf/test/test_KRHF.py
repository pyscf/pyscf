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
    basis  = basis,
    pseudo = pseudo,
    verbose = 10,
    ke_cutoff=70
)

nk = [2,2,2]  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
print("kpts = ", kpts)

kmf = scf.KRHF(cell, kpts)

dm = kmf.init_guess_by_1e()

nuc = kmf.get_hcore(kpts=kpts)

vj, vk = kmf.with_df.get_jk(dm, hermi=0)

C = 35
prim_partition = [[0,1,2,3,4,5,6,7]]
pbc_isdf_info  = isdf_linear_scaling_k.PBC_ISDF_Info_Quad_K(cell, kmesh=nk, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, rela_cutoff_QRCP=3e-3)
pbc_isdf_info.build_IP_local(c=C, m=5, group=prim_partition, Ls=[nk[0]*10, nk[1]*10, nk[2]*10])
pbc_isdf_info.build_auxiliary_Coulomb(debug=True)

vj2, vk2 = pbc_isdf_info.get_jk(dm, kpts = kpts)

#kmf.kernel()

print("vj  = ", vj[0, 0, :])
print("vj2 = ", vj2[0, 0, :])
print("vk  = ", vk[0, 0, :])
print("vk2 = ", vk2[0, 0, :])

#exit(1)

kmf.kernel()

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
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 
boxlen = 3.5668
prim_a = numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
KE_CUTOFF = 70
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
pbc_isdf_info.direct_scf = mf.direct_scf
mf.with_df = pbc_isdf_info
mf.max_cycle = 16
mf.conv_tol = 1e-7

dm = mf.init_guess_by_1e()

vj3, vk3 = mf.with_df.get_jk(dm, hermi=0)

print("vj3 = ", vj3[0, :])
print("vk3 = ", vk3[0, :])