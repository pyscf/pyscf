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

### todo write clustering code ### 



C = 10
M = 5

if __name__ == "__main__":
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    KE_CUTOFF = 256
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
    
    for i in range(cell.natm):
        print('%s %s  charge %f  xyz %s' % (cell.atom_symbol(i),
                                        cell.atom_pure_symbol(i),
                                        cell.atom_charge(i),
                                        cell.atom_coord(i)))

    print("Atoms' charges in a vector\n%s" % cell.atom_charges())
    print("Atoms' coordinates in an array\n%s" % cell.atom_coords())
    
    #### test 
    
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR)
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=True)
    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)
    
    V_R = pbc_isdf_info.V_R
    
    aux_bas = pbc_isdf_info.aux_basis
    
    
    # tmp = aux_bas[0:1, :].reshape(-1, *mesh)
    # tmp_HOSVD = HOSVD.HOSVD(tmp)
    
    # perform the compression row by row 
    
    for i in range(aux_bas.shape[0]):
        tmp = V_R[i:i+1, :].reshape(-1, *mesh)
        tmp_HOSVD = HOSVD.HOSVD(tmp, cutoff=1e-6)
        print("row i ", i, " tmp_HOSVD.Bshape = ", tmp_HOSVD.Bshape)
        # print(tmp_HOSVD.S)
        tmp_full = tmp_HOSVD.getFullMat()
        V_R[i:i+1, :] = tmp_full.reshape(1, -1)
    pbc_isdf_info.V_R = V_R
    
    print("mesh = ", mesh)
    
    # print("tmp_HOSVD.shape = ", tmp_HOSVD.shape)
    # print("tmp_HOSVD.Bshape = ", tmp_HOSVD.B.shape)
    # print("tmp_HOSVD.S = ", tmp_HOSVD.S)
    
    # tmp2 = V_R[0:1, :].reshape(-1, *mesh)
    # tmp_HOSVD = HOSVD.HOSVD(tmp2, cutoff=1e-8)
    

    from pyscf.pbc import scf
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7
    print("mf.direct_scf = ", mf.direct_scf)
    mf.kernel()
    
    