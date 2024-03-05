from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info
from pyscf.pbc.df.isdf.isdf_outcore import PBC_ISDF_Info_outcore

from functools import reduce
import numpy as np
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_ao2mo as isdf_ao2mo
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk

import pyscf.pbc.df.isdf.isdf_outcore as ISDF_Outcore 
import pyscf.pbc.df.isdf.isdf_k as ISDF_K
import pyscf.pbc.df.isdf.isdf_k_direct as ISDF_K_DIRECT

ISDF_K.COND_CUTOFF = 0.0 # not cutoff

from pyscf.lib.parameters import BOHR

MOL_STRUCTURE = '''
                   C     0.      0.      0.
                   C     0.8917  0.8917  0.8917
                   C     1.7834  1.7834  0.
                   C     2.6751  2.6751  0.8917
                   C     1.7834  0.      1.7834
                   C     2.6751  0.8917  2.6751
                   C     0.      1.7834  1.7834
                   C     0.8917  2.6751  2.6751
                '''
                
# C_ARRAY = [7, 15, 25, 35]
C_ARRAY = [15]
SuperCell_ARRAY = [
    [1, 1, 1],
    # [1, 1, 2],
    # [1, 2, 2],
    # [2, 2, 2],
    # [3, 3, 3],
    # [4, 2, 2],
    # [2, 2, 4],
    # [4, 4, 4],
    # [2, 2, 1],
    # [2, 2, 1],
    # [4, 4, 1],
    # [4, 4, 2],
]
# Ke_CUTOFF = [256, 512]
Ke_CUTOFF = [128]
# Ke_CUTOFF = [128]
boxlen = 3.5668
Basis = ['ecpccpvdz']

IO_MEMORY    = [int(2e9)]
BUNCHSIZE_IO = [32768]

if __name__ == '__main__':

    # boxlen = 3.5668
    prim_a = np.array(
                    [[14.572056092/2, 0.000000000, 0.000000000],
                     [0.000000000, 14.572056092/2, 0.000000000],
                     [0.000000000, 0.000000000,  6.010273939],]) * BOHR
    atm = [
['Cu1',	(1.927800,	1.927800,	1.590250)],
# ['Cu1',	(5.783400,	5.783400,	1.590250)],
# ['Cu2',	(1.927800,	5.783400,	1.590250)],
# ['Cu2',	(5.783400,	1.927800,	1.590250)],
# ['O1',	(1.927800,	3.855600,	1.590250)],
['O1',	(1.927800,	0.000000,	1.590250)],
# ['O1',	(3.855600,	5.783400,	1.590250)],
# ['O1',	(5.783400,	3.855600,	1.590250)],
# ['O1',	(3.855600,	1.927800,	1.590250)],
['O1',	(0.000000,	1.927800,	1.590250)],
# ['O1',	(1.927800,	7.711200,	1.590250)],
# ['O1',	(7.711200,	5.783400,	1.590250)],
# ['O1',	(5.783400,	0.000000,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
# ['Ca',	(3.855600,	3.855600,	0.000000)],
# ['Ca',	(7.711200,	3.855600,	0.000000)],
# ['Ca',	(3.855600,	7.711200,	0.000000)],
    ]
    
    basis = {
        'Cu1':'ecpccpvdz', 'Cu2':'ecpccpvdz', 'O1': 'ecpccpvdz', 'Ca':'ecpccpvdz'
    }
    
    pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
    
    for supercell in SuperCell_ARRAY:
        for ke_cutoff in Ke_CUTOFF:
            for basis in Basis:
                for c in C_ARRAY:
                    print('--------------------------------------------')
                    print('C = %d, supercell = %s, kc_cutoff = %d, basis = %s' % (
                        c, str(supercell), ke_cutoff, basis))

                    prim_cell = ISDF_K.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
                    prim_mesh = prim_cell.mesh
                    print("prim_mesh = ", prim_mesh)
            
                    mesh = [supercell[0] * prim_mesh[0], supercell[1] * prim_mesh[1], supercell[2] * prim_mesh[2]]
                    mesh = np.array(mesh, dtype=np.int32)
            
                    cell = ISDF_K.build_supercell(atm, prim_a, Ls = supercell, ke_cutoff=ke_cutoff, mesh=mesh, basis=basis, pseudo=pseudo)

                    # cell = pbcgto.Cell()
                    # cell.a = np.array(
                    #     [[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
                    # cell.atom = MOL_STRUCTURE
                    # cell.basis = basis
                    # # cell.pseudo = 'gth-pade'
                    # cell.pseudo = 'gth-hf'
                    # cell.verbose = 4
                    # cell.ke_cutoff = ke_cutoff
                    # cell.max_memory = 20000  # 20 GB
                    # cell.precision = 1e-8  # integral precision
                    # cell.use_particle_mesh_ewald = True
                    # cell.build(mesh=[27,27,27])
                    # cell = tools.super_cell(cell, supercell)
                    
                    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG
                    df_tmp = MultiGridFFTDF2(cell)

                    grids  = df_tmp.grids
                    coords = np.asarray(grids.coords).reshape(-1,3)
                    nx = grids.mesh[0]

                    mesh   = grids.mesh
                    ngrids = np.prod(mesh)
                    assert ngrids == coords.shape[0]

                    # aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T
                    # aoR  *= np.sqrt(cell.vol / ngrids)

                    for max_io_memory in IO_MEMORY:
                        # for bunch_size in BUNCHSIZE_IO:
                            
                        print("max_IO_memory = ", max_io_memory)
                            
                        # ISDF_Outcore.MAX_BUNCHSIZE = bunch_size
                        # print("IO_bunchsize = ", bunch_size)

                        t1 = (lib.logger.process_clock(),lib.logger.perf_counter())
                        # pbc_isdf_info = PBC_ISDF_Info(cell, aoR=aoR)
                        # pbc_isdf_info.build_IP_Sandeep(c=c)
                        # pbc_isdf_info.build_auxiliary_Coulomb(mesh=mesh)
                        pbc_isdf_info = ISDF_K_DIRECT.PBC_ISDF_Info_kSym_Direct(cell, max_io_memory, Ls=supercell)
                        pbc_isdf_info.build_kISDF_obj(c=c, m=5, global_selection=False)
                        # pbc_isdf_info.build_auxiliary_Coulomb()
                        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                        print(_benchmark_time(t1, t2, 'build_isdf'))

                        # for bunch_size in BUNCHSIZE_IO:

                        ### perform scf ###

                        from pyscf.pbc import scf

                        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        mf = scf.RHF(cell)
                        mf.with_df = pbc_isdf_info
                        mf.max_cycle = 64
                        mf.conv_tol = 1e-7
                        pbc_isdf_info.direct_scf = mf.direct_scf
                        mf.kernel()
                        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        print(_benchmark_time(t1, t2, 'scf_isdf'))

                        del pbc_isdf_info
                        pbc_isdf_info = None
                        mf = None
                    cell = None