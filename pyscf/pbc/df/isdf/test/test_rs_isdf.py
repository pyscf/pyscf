#!/usr/bin/env python

import numpy 
import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf import lib
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

KPTS = [
    [1,1,1],
    # [2,2,2],
    # [3,3,3],
    # [4,4,4],
]

# basis = 'unc-gth-cc-dzvp'
# pseudo = "gth-hf"  
# basis='6-31G'
# pseudo=None
basis = 'gth-dzvp'
pseudo = "gth-pade"  
ke_cutoff = 70  
    
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
    basis = '6-31g',
    verbose = 4,
)

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.
              C     0.8917  0.8917  0.8917''',
    basis = '6-31g',
    verbose = 4,
)


boxlen = 3.5668
prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
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
    
    omega = 0.8
    
    from pyscf.pbc.df.isdf.isdf_linear_scaling import PBC_ISDF_Info_Quad
    C = 12
    # group_partition = [[0,1],[2,3],[4,5],[6,7]]
    group_partition=[[0,1]]
    
    print("supercell.omega = ", supercell.omega)
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    pbc_isdf_info = PBC_ISDF_Info_Quad(supercell, with_robust_fitting=True, aoR_cutoff=1e-12, direct=True, omega=omega, rela_cutoff_QRCP=1e-5)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*3, Ls[1]*3, Ls[2]*3])
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    print("mesh = ", pbc_isdf_info.mesh)
    _benchmark_time(t1, t2, "build isdf")

    # supercell.omega = omega
    
    print("supercell.omega = ", supercell.omega)
    
    #
    # RS-JK builder is efficient for large number of k-points
    #
    mf = scf.RHF(supercell, kpts)
    mf.with_df = pbc_isdf_info
    # mf.kernel()
    
    # exit(1)
    
    dm = mf.get_init_guess(key='atom')
    
    '''
    ref =  -302.721144613077,  Full Electron 6-31G
    '''

    # continue

    print("-------------- Test RS-ISDF --------------")

    vj, vk = pbc_isdf_info.get_jk(dm, kpt=kpts)

    print("vj    = ", vj[0,:16])
    print("vj    = ", vj[-1,-16:])
    print("vk    = ", vk[0,:16])
    print("vk    = ", vk[-1,-16:])
    
    exit(1)
    
    print("-------------- Test RS-JK   --------------")
    
    supercell.omega = 0.0
    # print("supercell.omega = ", supercell.omega)
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.exclude_dd_block = False
    rsjk.allow_drv_nodddd = False
    rsjk.build(omega=omega)
    # print("rsjk has long range = ", rsjk.has_long_range())
    
    vj, vk = rsjk.get_jk(dm, kpts=kpts)
    
    print("vj    = ", vj[0,:16])
    print("vj    = ", vj[-1,-16:])
    print("vk    = ", vk[0,:16])
    print("vk    = ", vk[-1,-16:]) 
    
    # rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    # rsjk.exclude_dd_block = False
    # rsjk.allow_drv_nodddd = False
    # rsjk.build(omega=omega*2)
    # vj, vk = rsjk.get_jk(dm, kpts=kpts)
    # print("vj    = ", vj[0,:16])
    # print("vj    = ", vj[-1,-16:])
    # print("vk    = ", vk[0,:16])
    # print("vk    = ", vk[-1,-16:])
    
    print("-------------- Test RS-JK only LR  --------------")
    
    vj, vk = rsjk.get_jk(dm, kpts=kpts, omega=omega)
    print("vj_lr = ", vj[0,:16])
    print("vj_lr = ", vj[-1,-16:])
    print("vk_lr = ", vk[0,:16])
    print("vk_lr = ", vk[-1,-16:])
    
    # vj = rsjk._get_vj_lr(dm, kpts=kpts)
    # vk = rsjk._get_vk_lr(dm, kpts=kpts)
    
    # print("vj_lr = ", vj[0,:16])
    # print("vj_lr = ", vj[-1,-16:])
    # print("vk_lr = ", vk[0,:16])
    # print("vk_lr = ", vk[-1,-16:])
    
    print("-------------- Test RS-JK only SR  --------------")
    
    supercell.omega = -omega
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.exclude_dd_block = False
    rsjk.allow_drv_nodddd = False
    rsjk.build()
    print("rsjk has long range = ", rsjk.has_long_range())
    
    vj, vk = rsjk.get_jk(dm, kpts=kpts)
    
    print("vj_sr = ", vj[0,:16])
    print("vj_sr = ", vj[-1,-16:])
    print("vk_sr = ", vk[0,:16])
    print("vk_sr = ", vk[-1,-16:])
    
    # supercell.omega = -2.0
    # rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    # rsjk.build()
    # print("rsjk has long range = ", rsjk.has_long_range())