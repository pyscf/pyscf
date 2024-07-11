from pyscf.pbc.df.isdf.test.test_isdf_fast import PBC_ISDF_Info

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

C_ARRAY = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
SuperCell_ARRAY = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 2, 2],
    # [2, 2, 2],
    # [4, 2, 2],
]
Ke_CUTOFF = [256]
boxlen = 3.5668
Basis = ['gth-szv', 'gth-dzv', 'gth-dzvp', 'gth-tzvp']

if __name__ == '__main__':
    
    
    for supercell in SuperCell_ARRAY:
        for ke_cutoff in Ke_CUTOFF:
            for basis in Basis:
                for c in C_ARRAY[:1]:
                    print('--------------------------------------------')
                    print('C = %d, supercell = %s, kc_cutoff = %d, basis = %s' % (c, str(supercell), ke_cutoff, basis))

                    cell   = pbcgto.Cell()
                    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
                    cell.atom = MOL_STRUCTURE
                    cell.basis   = basis
                    cell.pseudo  = 'gth-pade'
                    cell.verbose = 4
                    cell.ke_cutoff = ke_cutoff
                    cell.max_memory = 20000  # 20 GB
                    cell.precision  = 1e-8  # integral precision
                    cell.use_particle_mesh_ewald = True
                    cell.build()

                    cell = tools.super_cell(cell, supercell)

                    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

                    df_tmp = MultiGridFFTDF2(cell)
                    grids  = df_tmp.grids
                    coords = np.asarray(grids.coords).reshape(-1,3)
                    mesh   = grids.mesh
                    ngrids = np.prod(mesh)
                    assert ngrids == coords.shape[0]

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
                    aoR  *= np.sqrt(cell.vol / ngrids)
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    print(_benchmark_time(t1, t2, 'eval_ao', cell))

                    print("aoR.shape = ", aoR.shape)

                    ## test the sparsity of aoR matrix
                    ## find the number of non-zero elements in aoR (abs > 1e-8)

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    n_nonzero = 0
                    for i in range(aoR.shape[0]):
                        non_zero_i = 0
                        for j in range(aoR.shape[1]):
                            if abs(aoR[i,j]) > 1e-8:
                                non_zero_i += 1
                        if non_zero_i > 0:
                            print("basis %3d has %10d non-zero elements" % (i, non_zero_i))
                            n_nonzero += non_zero_i
                    print("n_nonzero = ", n_nonzero)
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    print(_benchmark_time(t1, t2, 'count nonzero elements', cell))


                    # pair cutoff 

                    cutoff = 1e-10
                    # find the max abs value of aoR for each grid point 

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    max_abs_aoR = np.max(abs(aoR), axis = 0)
                    assert max_abs_aoR.shape[0] == ngrids
                    cutoff_on_grids = cutoff / max_abs_aoR

                    n_nonzero = 0
                    for i in range(aoR.shape[0]):
                        non_zero_i = 0
                        for j in range(aoR.shape[1]):
                            if abs(aoR[i,j]) > cutoff_on_grids[j]:
                                non_zero_i += 1
                        if non_zero_i > 0:
                            print("basis %3d has %10d non-zero elements" % (i, non_zero_i))
                            n_nonzero += non_zero_i
                    print("n_nonzero = ", n_nonzero)
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())