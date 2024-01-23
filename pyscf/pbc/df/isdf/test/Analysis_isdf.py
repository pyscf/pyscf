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

# C_ARRAY = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
C_ARRAY = [5]
SuperCell_ARRAY = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 2, 2],
    [2, 2, 2],
    # [4, 2, 2],
]
Ke_CUTOFF = [256]
boxlen = 3.5668
# Basis = ['gth-szv', 'gth-dzv', 'gth-dzvp', 'gth-tzvp']
Basis = ['gth-dzvp']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

NBASIS_SELECTED = 16

def _analysis_V_rank_block(V, row_partition:list[list], col_partition:list[list], threshold=1e-8):
    assert V.ndim == 2
    for id_row, row in enumerate(row_partition):
        V_tmp = V[row,:]
        for id_col, col in enumerate(col_partition):
            V_tmp2 = V_tmp[:, col]
            V_tmp3 = V_tmp2 @ V_tmp2.T
            # perform SVD
            e, h = np.linalg.eigh(V_tmp3)
            S = np.sqrt(e[::-1])
            # get the rank
            where = np.where(S > threshold)[0]
            rank = len(where)
            print("row = ", id_row, "col = ", id_col, "rank = ", rank)
            # print S with large value
            S = S[where]
            for i, s in enumerate(S):
                print("S[%3d] = %15.8e" % (i, s))

def _analysis_W_rank_block(W, partition:list[list], threshold=1e-8):
    assert W.ndim == 2
    for id_row, row in enumerate(partition):
        W_tmp = W[row,:]
        for id_col, col in enumerate(partition):
            W_tmp2 = W_tmp[:, col]
            # perform SVD
            e, h = np.linalg.eigh(W_tmp2)
            S = np.sqrt(e[::-1])
            # get the rank
            where = np.where(S > threshold)[0]
            rank = len(where)
            print("row = ", id_row, "col = ", id_col, "rank = ", rank)
            # print S with large value
            S = S[where]
            for i, s in enumerate(S):
                print("S[%3d] = %15.8e" % (i, s))

def _get_important_pnt(aux_basis:np.ndarray, weight, relative_cutoff=1e-3):
    assert aux_basis.ndim == 1

    aux_basis = np.abs(aux_basis)
    # get the norm
    aux_basis_norm = np.linalg.norm(aux_basis)
    print("weight         = ", weight)
    print("aux_basis_norm = ", aux_basis_norm * np.sqrt(weight))
    # get the cutoff
    cutoff = (1-relative_cutoff) * (aux_basis_norm**2)
    # print("cutoff = ", cutoff)
    # sort the aux_basis via abs, from large to small
    aux_basis_abs = np.abs(aux_basis)
    aux_basis_abs_sorted = np.sort(aux_basis_abs)[::-1]
    # print("aux_basis_abs_sorted = ", aux_basis_abs_sorted)
    # search the index below which the sum is smaller than cutoff
    aux_basis_abs_sorted_cumsum = np.cumsum(aux_basis_abs_sorted**2)
    idx = np.where(aux_basis_abs_sorted_cumsum < cutoff)[0]
    # print("idx = ", idx)
    # get the important points
    important_pnt = aux_basis_abs_sorted[idx[-1]]
    # print("important_pnt = ", important_pnt)
    # find the point with abs value largr than important_pnt
    idx = np.where(aux_basis_abs > important_pnt)[0]
    # print("idx = ", idx)
    # sort indx by the abs value of aux_basis
    idx = idx[np.argsort(aux_basis_abs[idx])[::-1]]
    return idx

if __name__ == '__main__':


    for supercell in SuperCell_ARRAY:
        for ke_cutoff in Ke_CUTOFF:
            for basis in Basis:
                for c in C_ARRAY:
                    print('--------------------------------------------')
                    print('C = %d, supercell = %s, kc_cutoff = %d, basis = %s' % (c, str(supercell), ke_cutoff, basis))

                    cell   = pbcgto.Cell()
                    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
                    cell.atom = MOL_STRUCTURE
                    cell.basis   = basis
                    cell.pseudo  = 'gth-pade'
                    cell.verbose = 1
                    cell.ke_cutoff = ke_cutoff
                    cell.max_memory = 20000  # 20 GB
                    cell.precision  = 1e-8  # integral precision
                    cell.use_particle_mesh_ewald = True
                    cell.build()

                    cell = tools.super_cell(cell, supercell)

                    atm_coord = cell.atom_coords()

                    print("atm_coord = ", atm_coord)

                    supercell_str = "%d%d%d" % (supercell[0], supercell[1], supercell[2])

                    # exit(1)

                    # get the coordinate of atm

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
                    print(_benchmark_time(t1, t2, 'eval_ao'))

                    print("aoR.shape = ", aoR.shape)

                    pbc_isdf_info = PBC_ISDF_Info(cell, aoR, cutoff_aoValue=1e-6, cutoff_QR=1e-3)
                    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=c, global_IP_selection=False)
                    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)

                    #### PRINT THE IP POINTS ####

                    IP_points = coords[pbc_isdf_info.IP_ID]

                    print("IP_points = ", IP_points)

                    IP_x = IP_points[:,0]
                    IP_y = IP_points[:,1]
                    IP_z = IP_points[:,2]

                    fig = plt.figure()
                    ax = Axes3D(fig)
                    ax.scatter(IP_x, IP_y, IP_z, c='blue', marker='o', s=60)

                    ax.scatter(atm_coord[:,0], atm_coord[:,1], atm_coord[:,2], c='r', marker='o', s=100)

                    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
                    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
                    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

                    plt.savefig("IP_points_%s.png" % supercell_str)
                    # plt.show()

                    #### random pick up 16 aux_basis and print the basis points ####

                    basis_indx = np.random.randint(0, pbc_isdf_info.aux_basis.shape[0], NBASIS_SELECTED)

                    for i, _id_ in enumerate(basis_indx):

                        print("basis %d" % _id_)
                        indx = _get_important_pnt(pbc_isdf_info.aux_basis[_id_], weight=cell.vol/ngrids)
                        print("we get %d important points" % len(indx))

                        for pnt in [1024, 2048, 4096]:

                            if pnt > len(indx):
                                continue

                            basis_coord = coords[indx[:pnt]]

                            basis_x = basis_coord[:,0]
                            basis_y = basis_coord[:,1]
                            basis_z = basis_coord[:,2]

                            fig = plt.figure()
                            ax = Axes3D(fig)
                            ax.scatter(basis_x, basis_y, basis_z, c='blue', marker='o', s=60)

                            IP_points = coords[pbc_isdf_info.IP_ID[_id_]]

                            IP_x = [IP_points[0]]
                            IP_y = [IP_points[1]]
                            IP_z = [IP_points[2]]

                            ax.scatter(IP_x, IP_y, IP_z, c='black', marker='x', s=500)

                            ax.scatter(atm_coord[:,0], atm_coord[:,1], atm_coord[:,2], c='r', marker='o', s=100)

                            ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
                            ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
                            ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

                            plt.savefig("basis_%d_%d_%s.png" % (i, pnt, supercell_str))
                            # plt.show()

                # explore the low-rank structure of the V and W matrix

                    voroini = pbc_isdf_info.partition
                    natm = cell.natm
                    grid_partition = []

                    for i in range(natm):
                        # get the indx of grid points in the voronoi cell of atom i
                        grid_partition.append(np.where(voroini == i)[0])

                    IP_partition = []

                    for i in range(natm):
                        # get the indx of IP points in the voronoi cell of atom i
                        tmp = [id for id, x in enumerate(pbc_isdf_info.IP_ID) if voroini[x] == i]
                        IP_partition.append(tmp)
                        # print("atom %d has %d IP points" % (i, len(tmp)))
                        # print("tmp = ", tmp)

                    print(" ************** analysis V ************** ")
                    _analysis_V_rank_block(pbc_isdf_info.V_R,IP_partition, grid_partition)
                    print(" ************** analysis W ************** ")
                    _analysis_W_rank_block(pbc_isdf_info.W, IP_partition)
