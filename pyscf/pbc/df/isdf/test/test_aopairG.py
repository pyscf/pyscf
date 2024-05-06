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
        ['C', (0.8917 , 0.8917 , 0.8917)],
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 

basis = 'unc-gth-dzvp'

for nk in KPTS:

    ke_cutoff = 32
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, verbose=4, pseudo=None)
    prim_mesh = prim_cell.mesh
    mesh = [nk[0] * prim_mesh[0], nk[1] * prim_mesh[1], nk[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    supercell = build_supercell(atm, prim_a, Ls = nk, ke_cutoff=None, basis=basis, verbose=4, pseudo=None, mesh=mesh)


    ### aftdf ###
    
    mesh = [5,5,5]
    mesh = np.array(mesh, dtype=np.int32)
    # aftdf2 = aft.AFTDF(supercell)
    Gv, Gvbase, kws = supercell.get_Gv_weights(mesh)
    aoPair_G = ft_ao.ft_aopair_kpts(supercell, Gv=Gv)
    print("aoPair_G.shape = ", aoPair_G.shape)
    aoPair_G = aoPair_G[0] 
    # print(aoPair_G[:, 10, 11])

    for i in range(aoPair_G.shape[1]):
        for j in range(aoPair_G.shape[2]):
            assert np.allclose(aoPair_G[:, i, j], aoPair_G[:, j, i])

    aoPair_G_imag = aoPair_G.imag
    
    print("max of imag = ", np.max(aoPair_G_imag))
    print("norm of imag = ", np.linalg.norm(aoPair_G_imag))

    aoPair_G = np.transpose(aoPair_G, axes=(1, 2, 0))
    aoPair_G = aoPair_G.reshape(aoPair_G.shape[0], aoPair_G.shape[1], *mesh)
    aoPair_R = np.fft.ifftn(aoPair_G, axes=(2, 3, 4))
    
    aoPair_R_imag = aoPair_R.imag
    aoPair_R = aoPair_R.real

    print("max of imag = ", np.max(aoPair_R_imag))
    print("norm of imag = ", np.linalg.norm(aoPair_R_imag))
    
    aoPair_R2 = np.fft.fftn(aoPair_G, axes=(2, 3, 4))
    aoPair_R2 = aoPair_R2.real
    aoPair_R2_imag = aoPair_R2.imag
    print("max of imag = ", np.max(aoPair_R2_imag))
    
    exit(1)