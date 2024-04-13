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

KPTS = [
    [1,1,1],
    # [2,2,2],
    # [3,3,3],
    # [4,4,4],
]

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

for nk in KPTS:

    # nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
    # kpts = cell.make_kpts(nk)

    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=None, basis=cell.basis, verbose=4, pseudo=None)
    prim_mesh = prim_cell.mesh
    mesh = [nk[0] * prim_mesh[0], nk[1] * prim_mesh[1], nk[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    supercell = build_supercell(atm, prim_a, Ls = nk, ke_cutoff=None, basis=cell.basis, verbose=4, pseudo=None, mesh=mesh)

    nk_supercell = [1,1,1]
    kpts = supercell.make_kpts(nk_supercell)

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
    
    supercell.omega = -1.0
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.build()
    print("rsjk has long range = ", rsjk.has_long_range())
    
    supercell.omega = -2.0
    rsjk = RangeSeparatedJKBuilder(supercell, kpts)
    rsjk.build()
    print("rsjk has long range = ", rsjk.has_long_range())
    