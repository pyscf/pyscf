import copy
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

from pyscf.pbc.df.isdf.isdf_k import build_supercell
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

from itertools import permutations

from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info
import pyscf.pbc.df.isdf.isdf_outcore as ISDF_outcore
import pyscf.pbc.df.isdf.isdf_fast as ISDF

import pyscf.pbc.df.isdf.test.test_HOSVD as HOSVD

import ctypes

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

from pyscf.lib.parameters import BOHR

if __name__ == "__main__":
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    KE_CUTOFF = 70
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
    
    # KE_CUTOFF = 128
    # prim_a = np.array(
    #                 [[14.572056092/2, 0.000000000, 0.000000000],
    #                  [0.000000000, 14.572056092/2, 0.000000000],
    #                  [0.000000000, 0.000000000,  6.010273939],]) * BOHR
    
#     atm = [
# ['Cu1',	(1.927800,	1.927800,	1.590250)],
# # ['Cu1',	(5.783400,	5.783400,	1.590250)],
# # ['Cu2',	(1.927800,	5.783400,	1.590250)],
# # ['Cu2',	(5.783400,	1.927800,	1.590250)],
# # ['O1',	(1.927800,	3.855600,	1.590250)],
# ['O1',	(1.927800,	0.000000,	1.590250)],
# # ['O1',	(3.855600,	5.783400,	1.590250)],
# # ['O1',	(5.783400,	3.855600,	1.590250)],
# # ['O1',	(3.855600,	1.927800,	1.590250)],
# ['O1',	(0.000000,	1.927800,	1.590250)],
# # ['O1',	(1.927800,	7.711200,	1.590250)],
# # ['O1',	(7.711200,	5.783400,	1.590250)],
# # ['O1',	(5.783400,	0.000000,	1.590250)],
# ['Ca',	(0.000000,	0.000000,	0.000000)],
# # ['Ca',	(3.855600,	3.855600,	0.000000)],
# # ['Ca',	(7.711200,	3.855600,	0.000000)],
# # ['Ca',	(3.855600,	7.711200,	0.000000)],
#     ]
    
    # basis = {
    #     'Cu1':'ecpccpvdz', 'Cu2':'ecpccpvdz', 'O1': 'ecpccpvdz', 'Ca':'ecpccpvdz'
    # }
    
    # pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    # prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF, basis=basis, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)
    
    Ls = [2, 1, 1]
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell = build_supercell(atm, prim_a, Ls=Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    # cell = build_supercell(atm, prim_a, Ls = Ls, ke_cutoff=KE_CUTOFF, basis=basis, pseudo=pseudo)
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)
    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    
    #### test 
    
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]
    
    bunchsize=1024
    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)
    
    # minimal = 1e10
    # maximal = -1e10
    
    # for p0, p1 in lib.prange(0, ngrids, bunchsize):
    #     aoR = df_tmp._numint.eval_ao(cell, coords[p0:p1])[0].T
    #     aoR *= np.sqrt(cell.vol / ngrids)
    #     max_aoR = np.max(np.abs(aoR), axis=0)
    
    # minimal = np.min(max_aoR)
    # maximal = np.max(max_aoR)
    # print("minimal = ", minimal)
    # print("maximal = ", maximal)
    
    max_aoR = np.max(np.abs(aoR), axis=0)
    
    where = np.where(max_aoR > 0.05)[0]
    print("where = ", where)
    print("len(where) = ", len(where))  
    
    ### plot histogram of max_aoR
    
    import matplotlib.pyplot as plt
    
    plt.hist(max_aoR, bins=100)
    plt.xscale('log')
    plt.show()
    
    ### plot histogram of one column of np.abs(aoR) with log scale
    
    plt.hist(np.abs(aoR[:,32]), bins=100)
    plt.xscale('log')
    plt.show()
    
    