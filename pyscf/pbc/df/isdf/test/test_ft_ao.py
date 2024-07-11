#!/usr/bin/env python

import numpy 
import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf import lib
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.ft_ao as ft_ao
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto 

KPTS = [
    [1,1,1],
    # [2,2,2],
    # [3,3,3],
    # [4,4,4],
]

basis = 'unc-gth-cc-dzvp'
pseudo = "gth-hf"  
# basis='6-31G'
# pseudo=None
ke_cutoff = 256
    
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
    basis = basis,
    verbose = 4,
)

boxlen = 3.5668
prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
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

for nk in KPTS:

    # nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
    # kpts = cell.make_kpts(nk)

    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, verbose=4, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    mesh = [nk[0] * prim_mesh[0], nk[1] * prim_mesh[1], nk[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    supercell = build_supercell(atm, prim_a, Ls = nk, ke_cutoff=ke_cutoff, basis=basis, verbose=4, pseudo=pseudo, mesh=mesh)

    nk_supercell = [1,1,1]
    kpts = supercell.make_kpts(nk_supercell)

    Ls = nk

    ######### test rs-isdf #########
    
    #omega = 5
    
    from pyscf.pbc.df.isdf.isdf_linear_scaling import PBC_ISDF_Info_Quad
    C = 10
    group_partition = [[0,1],[2,3],[4,5],[6,7]]
    
    #print("supercell.omega = ", supercell.omega)
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    pbc_isdf_info = PBC_ISDF_Info_Quad(supercell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, rela_cutoff_QRCP=1e-3)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*3, Ls[1]*3, Ls[2]*3])
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    print("mesh = ", pbc_isdf_info.mesh)
    _benchmark_time(t1, t2, "build isdf")

    Gv  = pbc_isdf_info.cell.get_Gv() 
    print("Gv.shape = ", Gv.shape)
    aoR = ISDF_eval_gto(pbc_isdf_info.cell, coords=pbc_isdf_info.coords) 
    aoR = aoR.reshape(-1, *pbc_isdf_info.cell.mesh)
    print("aoR.shape = ", aoR.shape)
    
    weight   = pbc_isdf_info.cell.vol/np.prod(pbc_isdf_info.cell.mesh)
    aoG_test = numpy.fft.fftn(aoR, axes=(1,2,3)).reshape(-1, np.prod(pbc_isdf_info.cell.mesh)) * weight
    aoG      = ft_ao.ft_ao(pbc_isdf_info.cell, Gv).T
        
    diff = np.linalg.norm(aoG_test-aoG)
    print("diff = ", diff/np.sqrt(np.prod(aoG.shape)))
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    aoR_test = numpy.fft.ifftn(aoG.reshape(-1, *pbc_isdf_info.cell.mesh), axes=(1,2,3)).real / (weight)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    print('time = ', t2[1]-t1[1], 's', 'time = ', t2[0]-t1[0], 's')
    
    diff = np.linalg.norm(aoR_test-aoR)
    print("diff = ", diff/np.sqrt(np.prod(aoR.shape)))  ### this can be extremely large for systems with core orbitals ! 