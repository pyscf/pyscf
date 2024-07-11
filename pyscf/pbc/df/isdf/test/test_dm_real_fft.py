#!/usr/bin/env python

from pyscf.pbc import gto, scf, dft
import numpy

import pyscf.pbc.df.isdf.isdf_linear_scaling_k as isdf_linear_scaling_k

basis  = 'gth-szv'
pseudo = 'gth-pade'

KE_CUTOFF = 70

cell = gto.M(
    a    = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917''',
    basis  = basis,
    pseudo = pseudo,
    verbose = 10,
    ke_cutoff=KE_CUTOFF
)

atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
    ] 

boxlen = 3.5668
prim_a = numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])

verbose = 10

from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell
from pyscf.pbc.df.isdf.isdf_linear_scaling_k import PBC_ISDF_Info_Quad_K 

prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], basis=basis, pseudo=pseudo, ke_cutoff=KE_CUTOFF)
prim_mesh = prim_cell.mesh

nk_list = [
    [1,1,1],
    [1,1,2],
    [1,1,3],
    [1,2,3],
    [1,3,3],
    [2,2,3],
    [2,3,3],
    [3,3,3]
]

for nk in nk_list:
    
    #nk = [2,2,3]  # 4 k-points for each axis, 4^3=64 kpts in total

    kpts = prim_cell.make_kpts(nk)
    print("nk   = ", nk)
    print("kpts = ", kpts)

    kmf = scf.KRHF(prim_cell, kpts)

    dm = kmf.init_guess_by_1e()

    dm_kpts = dm.copy()

    mesh = [prim_mesh[0]*nk[0], prim_mesh[1]*nk[1], prim_mesh[2]*nk[2]]
    cell = build_supercell(atm, prim_a, mesh=mesh, 
                                          Ls=nk,
                                       basis=basis, 
                                      pseudo=pseudo,
                                   ke_cutoff=KE_CUTOFF, 
                                     verbose=verbose)
    pbc_isdf_info = PBC_ISDF_Info_Quad_K(prim_cell, kmesh=nk, with_robust_fitting=False, aoR_cutoff=1e-8, direct=False, rela_cutoff_QRCP=3e-3)

    mf = scf.RHF(cell)
    mf.max_cycle = 16
    mf.conv_tol = 1e-7

    dm = mf.init_guess_by_1e()

    from pyscf.pbc.df.isdf.isdf_linear_scaling_k_jk import _preprocess_dm
    
    dm_real, _ = _preprocess_dm(pbc_isdf_info, dm_kpts)
    
    print("dm_real = ", dm_real.shape)
    print("dm      = ", dm.shape)
    diff_dm = numpy.linalg.norm(dm_real - dm) / numpy.sqrt(dm.size)
    print("diff_dm = ", diff_dm)
    
    assert diff_dm < 1e-7