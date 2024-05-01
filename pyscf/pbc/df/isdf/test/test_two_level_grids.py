#!/usr/bin/env python

import numpy 
import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf.lib.parameters import BOHR
from pyscf.pbc.df import aft, rsdf_builder, aft_jk, ft_ao
from pyscf import lib
from pyscf.pbc.df.isdf.isdf_tools_local import _estimate_rcut
import pyscf
import pyscf.pbc.gto as pbcgto

from pyscf.pbc.df.isdf import isdf_linear_scaling
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto
libpbc = lib.load_library('libpbc')
import ctypes, sys

if __name__ == "__main__":
    
    C = 10

    from pyscf.lib.parameters import BOHR
    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition

    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        #['C', (1.7834 , 1.7834 , 0.    )],
        #['C', (2.6751 , 2.6751 , 0.8917)],
        #['C', (1.7834 , 0.     , 1.7834)],
        #['C', (2.6751 , 0.8917 , 2.6751)],
        #['C', (0.     , 1.7834 , 1.7834)],
        #['C', (0.8917 , 2.6751 , 2.6751)],
    ] 
    KE_CUTOFF = 70
    
    basis = 'gth-dzvp'
    pseudo = "gth-pade"   
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF, basis=basis, pseudo=pseudo)    
    prim_partition = [[0,1]]
    verbose=4
    
    Ls = [1, 1, 1]
    
    mesh1 = [31,31,31]
    mesh1 = np.array(mesh1, dtype=np.int32) 
    cell1, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh1, 
                                                     Ls=Ls,
                                                     basis=basis, 
                                                     pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    weight1 = np.sqrt(cell1.vol / np.prod(mesh1))
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp  = MultiGridFFTDF2(cell1)
    grids   = df_tmp.grids
    coords1  = np.asarray(grids.coords).reshape(-1,3)
    assert coords1 is not None
    
    aoR1 = ISDF_eval_gto(cell1, coords=coords1) 
    aoR1 = aoR1.reshape(-1,*mesh1)
    print(aoR1.shape)
    aoR1_fft = np.fft.ifftn(aoR1, axes=(1,2,3)) * np.sqrt(cell1.vol)
    aoR1_fft = aoR1_fft.reshape(-1, np.prod(mesh1))
    
    mesh2 = [37,37,37]
    mesh2 = np.array(mesh2, dtype=np.int32) 
    cell2, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh2, 
                                                     Ls=Ls,
                                                     basis=basis, 
                                                     pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)

    weight2 = np.sqrt(cell2.vol / np.prod(mesh2))
    
    df_tmp  = MultiGridFFTDF2(cell2)
    grids   = df_tmp.grids
    coords2  = np.asarray(grids.coords).reshape(-1,3)
    assert coords2 is not None 
    
    aoR2 = ISDF_eval_gto(cell2, coords=coords2) 
    aoR2 = aoR2.reshape(-1,*mesh2)
    print(aoR2.shape)
    aoR2_fft = np.fft.ifftn(aoR2, axes=(1,2,3)) * np.sqrt(cell2.vol)
    aoR2_fft = aoR2_fft.reshape(-1, np.prod(mesh2))
    
    fn_map_fftfreq = getattr(libpbc, "map_fftfreq", None)
    assert(fn_map_fftfreq is not None)
    map1 = np.zeros(np.prod(mesh1), dtype=np.int32)
    
    fn_map_fftfreq(mesh1.ctypes.data_as(ctypes.c_void_p), mesh2.ctypes.data_as(ctypes.c_void_p), map1.ctypes.data_as(ctypes.c_void_p))

    print(map1)

    
    aoR1_fft2 = aoR2_fft[:,map1]
    
    print(aoR1_fft[0,:10])
    print(aoR1_fft2[0,:10])
    print(aoR1_fft[0,:10]/aoR1_fft2[0,:10])
    
    diff1 = np.linalg.norm(aoR1_fft - aoR1_fft2) / np.sqrt(aoR1_fft.size)
    print("diff1 = ", diff1)
    print("max = ", np.max(np.abs(aoR1_fft - aoR1_fft2)))
    
    
    fn_map_rfftfreq = getattr(libpbc, "map_rfftfreq", None)
    assert fn_map_rfftfreq is not None
    size1 = mesh1[0] * mesh1[1] * (mesh1[2]//2+1)
    map2 = np.zeros(size1, dtype=np.int32)
    
    fn_map_rfftfreq(mesh1.ctypes.data_as(ctypes.c_void_p), mesh2.ctypes.data_as(ctypes.c_void_p), map2.ctypes.data_as(ctypes.c_void_p)) 
    
    print(map2)
    
    aoR1_rfft = np.fft.rfftn(aoR1, axes=(1,2,3)) * np.sqrt(cell1.vol) / np.prod(mesh1)
    aoR1_rfft = aoR1_rfft.reshape(-1, size1)
    size2 = mesh2[0] * mesh2[1] * (mesh2[2]//2+1)
    aoR2_rfft = np.fft.rfftn(aoR2, axes=(1,2,3)) * np.sqrt(cell2.vol) / np.prod(mesh2)
    aoR2_rfft = aoR2_rfft.reshape(-1, size2)
    
    aoR1_rfft2 = aoR2_rfft[:,map2]
    
    print(aoR1_rfft[0,:10])
    print(aoR1_rfft2[0,:10])
    print(aoR1_rfft[0,:10]/aoR1_rfft2[0,:10])
    
    diff2 = np.linalg.norm(aoR1_rfft - aoR1_rfft2) / np.sqrt(aoR1_rfft.size)
    print("diff2 = ", diff2)
    print("max = ", np.max(np.abs(aoR1_rfft - aoR1_rfft2)))
    