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

########## helper function ##########

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
    # [2,2,2],
    # [3,3,3],
    # [4,4,4],
]

prim_a = np.array(
                    [[14.572056092, 0.000000000, 0.000000000],
                     [0.000000000, 14.572056092, 0.000000000],
                     [0.000000000, 0.000000000,  6.010273939],]) * BOHR
atm = [
['Cu1',	(1.927800,	1.927800,	1.590250)],
['Cu1',	(5.783400,	5.783400,	1.590250)],
['Cu2',	(1.927800,	5.783400,	1.590250)],
['Cu2',	(5.783400,	1.927800,	1.590250)],
['O1',	(1.927800,	3.855600,	1.590250)],
['O1',	(1.927800,	0.000000,	1.590250)],
['O1',	(3.855600,	5.783400,	1.590250)],
['O1',	(5.783400,	3.855600,	1.590250)],
['O1',	(3.855600,	1.927800,	1.590250)],
['O1',	(0.000000,	1.927800,	1.590250)],
['O1',	(1.927800,	7.711200,	1.590250)],
['O1',	(7.711200,	5.783400,	1.590250)],
['O1',	(5.783400,	0.000000,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
['Ca',	(3.855600,	3.855600,	0.000000)],
['Ca',	(7.711200,	3.855600,	0.000000)],
['Ca',	(3.855600,	7.711200,	0.000000)],
]

basis = 'unc-ecpccpvdz'

for nk in KPTS:

    # nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
    # kpts = cell.make_kpts(nk)
    ke_cutoff = 32
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, verbose=4, pseudo=None)
    prim_mesh = prim_cell.mesh
    mesh = [nk[0] * prim_mesh[0], nk[1] * prim_mesh[1], nk[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    supercell = build_supercell(atm, prim_a, Ls = nk, ke_cutoff=None, basis=basis, verbose=4, pseudo=None, mesh=mesh)

    nk_supercell = [1,1,1]
    kpts = supercell.make_kpts(nk_supercell)

    rcut = _estimate_rcut(supercell, np.prod(mesh), 1e-8)
    print("rcut for ovlp = ", rcut)

    #
    # RS-JK builder is efficient for large number of k-points
    #
    # kmf = scf.KRHF(supercell, kpts).jk_method('RS')
    # kmf.kernel()

    # supercell.omega = -1.0
    print("supercell.omega = ", supercell.omega)
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.build(omega=1.4)
    print("rsjk has long range = ", rsjk.has_long_range())
    
    print("original cell bas info")
    print_bas_info(supercell)

    print("rs cell       bas info")
    print_bas_info(rsjk.rs_cell)
    
    print("cell_d        bas info")
    print_bas_info(rsjk.cell_d)
    
    # print("supmol_ft     bas info")
    # print_bas_info(rsjk.supmol_ft)
    
    cell_c = rsjk.rs_cell.compact_basis_cell()
    print("cell_c        bas info")
    print_bas_info(cell_c)
    print("rcut = ", cell_c.rcut)
    rcut = _estimate_rcut(cell_c, np.prod(cell_c.mesh), 1e-8)
    print("rcut for ovlp = ", rcut)
    cell_c = pyscf.pbc.df.ft_ao._RangeSeparatedCell.from_cell(cell_c, rsjk.ke_cutoff, in_rsjk=True)
    
    # rcut = rsdf_builder.estimate_ft_rcut(rs_cell, exclude_dd_block=self.exclude_dd_block)
    supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(cell_c, [1,1,1], rcut.max())
    supmol_ft = supmol_ft.strip_basis(rcut)
    
    print_bas_info(supmol_ft)
    
    mesh = [2, 2, 2]
    cell_c.build(mesh=mesh)
    aosym   = 's1'
    ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=True, kpts=kpts)
    
    ngrids = np.prod(mesh)
    Gv, Gvbase, kws = cell_c.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    print("Gv     = ", Gv)
    print("Gvbase = ", Gvbase)
    print("gxyz   = ", gxyz)
    
    print("supmol_ft nao = ", supmol_ft.nao_nr())
    print("cell_c    nao = ", cell_c.nao_nr())
    kpt_allow = np.zeros(3)
    kpts = np.zeros((1,3))
    # Gblksize = 16
    Gblksize = ngrids
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts, None)
        print("Gpq.shape = ", Gpq.shape)
    
    Gpq = Gpq[0]
    print("Gpq.shape = ", Gpq.shape)
    Gpq = np.transpose(Gpq, (1,2,0))
    print("Gpq.shape = ", Gpq.shape)
    Gpq = Gpq.reshape(*Gpq.shape[:2], *mesh)
    print("Gpq.shape = ", Gpq.shape)
    print("Gpq[0,0] = ", Gpq[0,0])
    
    Gpq_test = Gpq[0,0]
    print("Gpq_test[0,0,0] = ", Gpq_test[0,0,0])
    print("Gpq_test[0,0,1] = ", Gpq_test[0,0,1])
    print("Gpq_test[0,1,0] = ", Gpq_test[0,1,0])    
    print("Gpq_test[0,1,1] = ", Gpq_test[0,1,1])
    print("Gpq_test[1,0,0] = ", Gpq_test[1,0,0])
    print("Gpq_test[1,0,1] = ", Gpq_test[1,0,1])
    print("Gpq_test[1,1,0] = ", Gpq_test[1,1,0])
    print("Gpq_test[1,1,1] = ", Gpq_test[1,1,1])
    
    Gpq2 = ft_ao.ft_aopair(supercell, Gv=Gv)
    Gpq2 = np.transpose(Gpq2, (1,2,0))
    print("Gpq2.shape = ", Gpq2.shape)
    print("Gpq2[0,0] = ", Gpq2[0,0])
    Gpq_test = Gpq2[0,0].reshape(*mesh)
    print("Gpq_test[0,0,0] = ", Gpq_test[0,0,0])
    print("Gpq_test[0,0,1] = ", Gpq_test[0,0,1])
    print("Gpq_test[0,1,0] = ", Gpq_test[0,1,0])    
    print("Gpq_test[0,1,1] = ", Gpq_test[0,1,1])
    print("Gpq_test[1,0,0] = ", Gpq_test[1,0,0])
    print("Gpq_test[1,0,1] = ", Gpq_test[1,0,1])
    print("Gpq_test[1,1,0] = ", Gpq_test[1,1,0])
    print("Gpq_test[1,1,1] = ", Gpq_test[1,1,1]) ### 错个相位 ? 
    
    exit(1)
    
    Gpq = np.fft.ifftn(Gpq, axes=(2,3,4))
    # Gpq = np.fft.fftn(Gpq, axes=(2,3,4))
    print("Gpq.shape = ", Gpq.shape)
    print("Gpq[0,0] = ", Gpq[0,0])
    Gpq_IMAG = Gpq.imag # check why it is not zero 
    print("np.max(np.abs(Gpq_IMAG)) = ", np.max(np.abs(Gpq_IMAG)))
    Gpq = Gpq.real
    # assert np.allclose(Gpq_IMAG, 0.0, atol=1e-6)
    Gpq = Gpq.reshape(*Gpq.shape[:2], -1)
    
    nao = cell_c.nao_nr()
    
    nImportantPair = 0
    ndataImportant = 0
    atm_connected = np.zeros((cell_c.natm, cell_c.natm), dtype=bool)
    ao_2_atm = get_ao_2_atm(cell_c) 
    print("ao_2_atm = ", ao_2_atm)
    for i in range(nao):
        for j in range(nao):
            value = Gpq[i,j]
            if np.max(np.abs(value)) > 1e-8:
                # print("i = %d, j = %d, value = " % (i, j), np.max(np.abs(value)))
                nImportantPair += 1
                ndataImportant += len(np.where(np.abs(value) > 1e-8)[0])
                atm_connected[ao_2_atm[i], ao_2_atm[j]] += 1
                
    print("nImportantPair = ", nImportantPair)
    print("ndataImportant = ", ndataImportant)
    
    # for i in range(cell_c.natm):
    #     for j in range(cell_c.natm):
    #         if atm_connected[i,j] > 0:
    #             print("atm %d is connected with atm %d" % (i, j))
    
    # fft 
        
    #### generate #### 
    
    continue
    
    supercell.omega = -1.0
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.build()
    print("rsjk has long range = ", rsjk.has_long_range())
    
    supercell.omega = -2.0
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.build()
    print("rsjk has long range = ", rsjk.has_long_range())
    