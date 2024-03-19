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
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)
    
    Ls = [1, 1, 1]
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell = build_supercell(atm, prim_a, Ls=Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)
    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    
    #### test 
    
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]
    
    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)
    
    nao_atm = int(cell.nao_nr() / cell.natm)
    
    print("nao_atm = ", nao_atm)
    
    for i in range(cell.natm):
    # for i in range(cell.nao):
        print("i = ", i)
        print("aoR.shape = ", aoR.shape)
        
        aoR_atm = aoR[i*nao_atm:(i+1)*nao_atm, :]
        # aoR_atm = aoR[i:(i+1), :]
        
        aoPairR = np.einsum('ip,jp->ijp', aoR_atm, aoR) 
        
        print("aoPairR.shape = ", aoPairR.shape)
        
        aoPairR = aoPairR.reshape(nao_atm, cell.nao_nr(), *mesh)
        
        print("aoPairR.shape = ", aoPairR.shape)
        
        Res = HOSVD.HOSVD_5D(aoPairR, cutoff=0.0, rela_cutoff=1e-10)
        
        print(Res['S'][0])
        print(Res['S'][1])
        print(Res['S'][2])
        print(Res['S'][3])
        print(Res['S'][4])
        