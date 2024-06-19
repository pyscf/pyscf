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
ke_cutoff = 128 
prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo, spin=1, verbose=10)
prim_mesh = prim_cell.mesh
KE_CUTOFF = 128
prim_partition = [[0, 1, 2, 3]]

mf = scf.UHF(prim_cell)
mf.max_cycle = 12
#mf.kernel()

###### test isdf ######


from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

df_tmp = MultiGridFFTDF2(prim_cell)

grids  = df_tmp.grids
coords = np.asarray(grids.coords).reshape(-1,3)
nx     = grids.mesh[0]

mesh   = grids.mesh
ngrids = np.prod(mesh)
assert ngrids == coords.shape[0]

aoR   = df_tmp._numint.eval_ao(prim_cell, coords)[0].T  # the T is important
aoR  *= np.sqrt(prim_cell.vol / ngrids)

C = 15
pbc_isdf_info = isdf_fast.PBC_ISDF_Info(prim_cell, aoR)
pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=False)
pbc_isdf_info.build_auxiliary_Coulomb(prim_cell, mesh) 

mf = scf.UHF(prim_cell)
pbc_isdf_info.direct_scf = mf.direct_scf
mf.with_df = pbc_isdf_info
mf.max_cycle = 100
mf.conv_tol = 1e-7
mf.max_cycle = 16
mf.kernel()

###### test isdf ######

Ls = [1,1,1]

group_partition = [[0],[1],[2],[3]]

pbc_isdf_info = isdf_linear_scaling.PBC_ISDF_Info_Quad(prim_cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=True, use_occ_RI_K=False,
                                                       rela_cutoff_QRCP=2e-4)
pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
pbc_isdf_info.Ls = Ls
pbc_isdf_info.build_auxiliary_Coulomb(debug=True) 

mf = scf.UHF(prim_cell)
pbc_isdf_info.direct_scf = mf.direct_scf
mf.with_df = pbc_isdf_info
mf.max_cycle = 100
mf.conv_tol = 1e-7
mf.max_cycle = 16
mf.kernel()

