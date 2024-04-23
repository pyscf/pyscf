#!/usr/bin/env python

'''
Hartree-Fock/DFT with k-points sampling for all-electron calculations

GDF (Gaussian density fitting), MDF (mixed density fitting), RSGDF
(range-separated Gaussian density fitting), or RS-JK builder
can be used in all electron calculations. They are more efficient than the
default SCF JK builder.
'''



import numpy 
import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf.lib.parameters import BOHR

from pyscf.pbc.df.isdf.isdf_tools_local import _estimate_rcut

########## helper function ##########

def print_bas_info(cell):
    for i in range(cell.nbas):
        print('shell %d on atom %d l = %s has %d contracted GTOs' %
            (i, cell.bas_atom(i), cell.bas_angular(i), cell.bas_nctr(i)))

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
    rsjk.build()
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
    
    continue
    
    supercell.omega = -1.0
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.build()
    print("rsjk has long range = ", rsjk.has_long_range())
    
    supercell.omega = -2.0
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.build()
    print("rsjk has long range = ", rsjk.has_long_range())
    