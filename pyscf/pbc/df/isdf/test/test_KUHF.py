#!/usr/bin/env python

'''
Mean field with k-points sampling

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.
'''

from pyscf.pbc import gto, scf, dft
import numpy as np
from pyscf.lib.parameters import BOHR
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell
import pyscf.pbc.df.isdf.isdf_fast as isdf_fast
import pyscf.pbc.df.isdf.isdf_linear_scaling as isdf_linear_scaling
import pyscf.pbc.df.isdf.isdf_linear_scaling_k as isdf_linear_scaling_k

prim_a = np.array(
                [[14.572056092/2, 0.000000000, 0.000000000],
                 [0.000000000, 14.572056092/2, 0.000000000],
                 [0.000000000, 0.000000000,  6.010273939],]) * BOHR
atm = [
['Cu1',	(1.927800,	1.927800,	1.590250)],
['O1',	(1.927800,	0.000000,	1.590250)],
['O1',	(0.000000,	1.927800,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
]
basis = {
    'Cu1':'ecpccpvdz', 'Cu2':'ecpccpvdz', 'O1': 'ecpccpvdz', 'Ca':'ecpccpvdz'
}
pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
ke_cutoff = 70 
prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo, spin=1, verbose=10)
prim_mesh = prim_cell.mesh
KE_CUTOFF = 70
prim_partition = [[0, 1, 2, 3]]

nk = [1, 1, 4]  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = prim_cell.make_kpts(nk)

mf = scf.KUHF(prim_cell, kpts)
mf.max_cycle = 4
#mf.kernel()
#print(mf.mo_occ)


print(mf.istype("UHF"))   ## FALSE
print(mf.istype("KUHF"))  ## TRUE
#exit(1)

###### test isdf ######

from pyscf.pbc.scf.addons import pbc_frac_occ

###### test isdf ######

#Ls = [1,1,1]
Ls = nk
group_partition = [[0],[1],[2],[3]]

pbc_isdf_info = isdf_linear_scaling_k.PBC_ISDF_Info_Quad_K(
                                                    prim_cell, 
                                                    with_robust_fitting=True, 
                                                    aoR_cutoff=1e-8, 
                                                    #direct=True, 
                                                    #use_occ_RI_K=False,
                                                    kmesh=nk,
                                                    rela_cutoff_QRCP=2e-4)
C=25
pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
pbc_isdf_info.build_auxiliary_Coulomb(debug=True) 

#mf = scf.UHF(prim_cell)
prim_cell.spin = 0
mf = scf.KUHF(prim_cell, kpts)
#mf = pbc_frac_occ(mf,tol=2e-4)
mf = scf.addons.smearing_(mf, 0.01) # the solution to the breakdown of TR symmetry 
pbc_isdf_info.direct_scf = mf.direct_scf
mf.with_df = pbc_isdf_info
mf.max_cycle = 100
mf.conv_tol = 1e-7
mf.max_cycle = 32
mf.kernel()

print(mf.mo_occ)

