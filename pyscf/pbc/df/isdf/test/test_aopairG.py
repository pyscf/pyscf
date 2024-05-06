#!/usr/bin/env python

import numpy 
import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf.lib.parameters import BOHR
from pyscf.pbc.df import aft, rsdf_builder, aft_jk, ft_ao
from pyscf import lib
from pyscf.pbc.df.isdf.isdf_tools_local import _estimate_rcut
import pyscf
import pyscf.pbc.gto as pbcgto

########## helper function ##########

def build_supercell(prim_atm, 
                    prim_a, 
                    mesh=None, 
                    Ls = [1,1,1], 
                    basis='gth-dzvp', 
                    pseudo='gth-pade', 
                    ke_cutoff=70, 
                    max_memory=2000, 
                    precision=1e-8,
                    use_particle_mesh_ewald=True,
                    verbose=4):
    
    Cell = pbcgto.Cell()
    
    assert prim_a[0, 1] == 0.0
    assert prim_a[0, 2] == 0.0
    assert prim_a[1, 0] == 0.0
    assert prim_a[1, 2] == 0.0
    assert prim_a[2, 0] == 0.0
    assert prim_a[2, 1] == 0.0
    
    Supercell_a = prim_a * np.array(Ls)
    Cell.a = Supercell_a
    
    atm = []
    
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                shift = [ix * prim_a[0, 0], iy * prim_a[1, 1], iz * prim_a[2, 2]]
                for atom in prim_atm:
                    atm.append([atom[0], (atom[1][0] + shift[0], atom[1][1] + shift[1], atom[1][2] + shift[2])])
    
    Cell.atom = atm
    Cell.basis = basis
    Cell.pseudo = pseudo
    Cell.ke_cutoff = ke_cutoff
    Cell.max_memory = max_memory
    Cell.precision = precision
    Cell.use_particle_mesh_ewald = use_particle_mesh_ewald
    Cell.verbose = verbose
    Cell.unit = 'angstorm'
    
    Cell.build(mesh=mesh)
    
    return Cell

def print_bas_info(cell):
    for i in range(cell.nbas):
        print('shell %d on atom %d l = %s has %d contracted GTOs' %
            (i, cell.bas_atom(i), cell.bas_angular(i), cell.bas_nctr(i)))

def get_ao_2_atm(cell):
    bas_2_atm = []
    for i in range(cell.nbas):
        # bas_2_atm += [cell.bas_atom(i)] * cell.bas_nctr(i)
        bas_2_atm.extend([cell.bas_atom(i)] * cell.bas_nctr(i)*(2*cell.bas_angular(i)+1))
    return bas_2_atm

KPTS = [
    [1,1,1],
]

boxlen = 3.5668
prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
atm = [
        ['C', (0.     , 0.     , 0.    )],
        # ['C', (0.8917 , 0.8917 , 0.8917)],
        # ['C', (1.7834 , 1.7834 , 0.    )],
        # ['C', (2.6751 , 2.6751 , 0.8917)],
        # ['C', (1.7834 , 0.     , 1.7834)],
        # ['C', (2.6751 , 0.8917 , 2.6751)],
        # ['C', (0.     , 1.7834 , 1.7834)],
        # ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 

basis = 'gth-dzvp'

for nk in KPTS:

    ke_cutoff = 256
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, verbose=4, pseudo=None)
    prim_mesh = prim_cell.mesh
    mesh = [nk[0] * prim_mesh[0], nk[1] * prim_mesh[1], nk[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    supercell = build_supercell(atm, prim_a, Ls = nk, ke_cutoff=None, basis=basis, verbose=4, pseudo=None, mesh=mesh)

    ### aftdf ###
    
    mesh = supercell.mesh
    mesh = np.array(mesh, dtype=np.int32)
    Gv, Gvbase, kws = supercell.get_Gv_weights(mesh)
    aoPair_G = ft_ao.ft_aopair_kpts(supercell, Gv=Gv)
    print("aoPair_G.shape = ", aoPair_G.shape)
    aoPair_G = aoPair_G[0] 

    for i in range(aoPair_G.shape[1]):
        for j in range(aoPair_G.shape[2]):
            assert np.allclose(aoPair_G[:, i, j], aoPair_G[:, j, i])

    aoPair_G_imag = aoPair_G.imag
    
    print("max of imag = ", np.max(aoPair_G_imag))
    print("norm of imag = ", np.linalg.norm(aoPair_G_imag))

    aoPair_G = np.transpose(aoPair_G, axes=(1, 2, 0))
    aoPair_G = aoPair_G.reshape(aoPair_G.shape[0], aoPair_G.shape[1], *mesh)
    aoPair_R = np.fft.ifftn(aoPair_G, axes=(2, 3, 4)) ### this is the benchmark!
    
    aoPair_R_imag = aoPair_R.imag
    aoPair_R      = aoPair_R.real
    aoPair_R = aoPair_R.reshape(aoPair_R.shape[0], aoPair_R.shape[1], -1)

    print("max of imag = ", np.max(aoPair_R_imag))
    print("norm of imag = ", np.linalg.norm(aoPair_R_imag))
    
    ##### check aoPair_R #####
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp = MultiGridFFTDF2(supercell)
    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(supercell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(supercell.vol / ngrids)

    aoPair_benchmark = np.einsum('ik,jk->ijk', aoR, aoR)
    
    print("aoPair_R.shape = ", aoPair_R.shape)
    print(aoPair_benchmark[0,0,:10])
    print(aoPair_R[0,0,:10])
    print(aoPair_benchmark[0,0,:10]/(aoPair_R[0,0,:10]))
    
    diff = np.linalg.norm(aoPair_R - aoPair_benchmark) 
    print("diff = ", diff/np.sqrt(aoPair_R.size)) # numerical they are the same, as kecutoff is large enough, the diff should decrease
    
    