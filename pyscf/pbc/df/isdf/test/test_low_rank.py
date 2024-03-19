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

def _distance_translation(pa:np.ndarray, pb:np.ndarray, a):
    '''
    calculate the distance between pa pb, but taken the periodic boundary condition into account
    '''
    # print("a = ", a)
    
    dx = pa[0] - pb[0]
    dx1 = dx - a[0][0]
    dx2 = dx + a[0][0]
    dx = abs(dx)
    dx1 = abs(dx1)
    dx2 = abs(dx2)
    dx = min(dx, dx1, dx2)
    
    dy = pa[1] - pb[1]
    dy1 = dy - a[1][1]
    dy2 = dy + a[1][1]
    dy = abs(dy)
    dy1 = abs(dy1)
    dy2 = abs(dy2)
    dy = min(dy, dy1, dy2)
    
    dz = pa[2] - pb[2]
    dz1 = dz - a[2][2]
    dz2 = dz + a[2][2]
    dz = abs(dz)
    dz1 = abs(dz1)
    dz2 = abs(dz2)
    dz = min(dz, dz1, dz2)
    
    return np.sqrt(dx**2 + dy**2 + dz**2)
    
def _middle_pnt(pa:np.ndarray, pb:np.ndarray, a):
    '''
    calculate the middle point between pa and pb, but taken the periodic boundary condition into account
    '''
    
    dx = pa[0] - pb[0]
    dx1 = dx - a[0][0]
    dx2 = dx + a[0][0]
    dx = abs(dx)
    dx1 = abs(dx1)
    dx2 = abs(dx2)
    dx_min = min(dx, dx1, dx2)
    
    mid_x = 0
    
    if dx_min == dx:
        mid_x = (pa[0] + pb[0]) / 2
    else:
        if dx_min == dx1:
            mid_x = (pa[0] + pb[0] - a[0][0]) / 2
        else:
            mid_x = (pa[0] + pb[0] + a[0][0]) / 2
    
    dy = pa[1] - pb[1]
    dy1 = dy - a[1][1]
    dy2 = dy + a[1][1]
    dy = abs(dy)
    dy1 = abs(dy1)
    dy2 = abs(dy2)
    dy_min = min(dy, dy1, dy2)
    
    mid_y = 0
    
    if dy_min == dy:
        mid_y = (pa[1] + pb[1]) / 2
    else:
        if dy_min == dy1:
            mid_y = (pa[1] + pb[1] - a[1][1]) / 2
        else:
            mid_y = (pa[1] + pb[1] + a[1][1]) / 2
    
    dz = pa[2] - pb[2]
    dz1 = dz - a[2][2]
    dz2 = dz + a[2][2]
    dz = abs(dz)
    dz1 = abs(dz1)
    dz2 = abs(dz2)
    dz_min = min(dz, dz1, dz2)
    
    mid_z = 0
    
    if dz_min == dz:
        mid_z = (pa[2] + pb[2]) / 2
    else:
        if dz_min == dz1:
            mid_z = (pa[2] + pb[2] - a[2][2]) / 2
        else:
            mid_z = (pa[2] + pb[2] + a[2][2]) / 2
    
    return np.array([mid_x, mid_y, mid_z])
    
def _cluster_IP(cell: Cell, Connection_Info:dict, partition, IP_ID, coords):
    atm_symbol = []
    atm_pure_symbol = []
    atm_coord = []
    natm = cell.natm
    distance_matrix = np.zeros((natm, natm))
    
    a = cell.lattice_vectors()
    
    # print("a = ", a)
    
    bond_mid_coord = {}
    atm_connection = []
    
    for i in range(natm):
        atm_symbol.append(cell.atom_symbol(i))
        atm_pure_symbol.append(cell.atom_pure_symbol(i))
        atm_coord.append(cell.atom_coord(i))
    for i in range(natm):
        for j in range(natm):
            distance_matrix[i,j] = _distance_translation(atm_coord[i], atm_coord[j], a)

    # print("distance_matrix = ", distance_matrix)    

    for i in range(natm):
        atm_connected_id = []
        for atm_label in Connection_Info[atm_symbol[i]]:
            natm_connected = Connection_Info[atm_symbol[i]][atm_label]
            where = np.where(np.array(atm_symbol) == atm_label)[0]
            distance = distance_matrix[i, where]
            ## find the natm_connected-th cloest atom to atm_label ##
            idx = np.argsort(distance)
            # print("idx = ", idx)
            # print("where = ", where)
            if abs(distance_matrix[i, where[idx[0]]]) < 1e-8: # the same atm
                atm_connected_id_tmp = where[idx[1:natm_connected+1]]
            else:
                atm_connected_id_tmp = where[idx[:natm_connected]]
            # print("atm_connected_id_tmp = ", atm_connected_id_tmp)
            atm_connected_id.extend(atm_connected_id_tmp)
        # for each atm, we have a list of connected atm id
        atm_connection.append(atm_connected_id)
        for j in atm_connected_id:
            if i < j:
                bond_mid_coord[(i,j)] = _middle_pnt(atm_coord[i], atm_coord[j], a)
                # print("bond_mid_coord[(i,j)] = ", bond_mid_coord[(i,j)])    

    ## first group to be the same atom type ## 
    
    IP_atm_group = []
    IP_cluster = {}    
    
    for i in range(natm):
        IP_atm_group.append([])
        IP_cluster[i] = []
    
    for key in bond_mid_coord:
        IP_cluster[key] = []
    
    for ip_id in IP_ID:
        atm_id = partition[ip_id]
        IP_atm_group[atm_id].append(ip_id)
    
    # for each ip, test whether it belongs to the cluster of the bond_mid_coord
    
    for i in range(natm):
        
        IP_cluster_now   = IP_atm_group[i]
        atm_i_coord      = atm_coord[i]
        connected_atm_id = atm_connection[i]

        for ip_id in IP_atm_group[i]:
            
            atm_id_belong = i 
            # print("ip_id = ", ip_id)
            # print("atm_id_belong = ", atm_id_belong)
            # print("atm_i_coord = ", atm_i_coord)
            # print("coords[ip_id] = ", coords[ip_id])
            dist_now      = _distance_translation(atm_i_coord, coords[ip_id], a)
            
            # print("dist_now = ", dist_now)
            
            for j in connected_atm_id:
                pair      = (min(i,j), max(i,j))
                coord_mid = bond_mid_coord[pair]
                dist_mid  = _distance_translation(coord_mid, coords[ip_id], a)
                # print("dist_mid = ", dist_mid)  
                if dist_mid < dist_now:
                    atm_id_belong = j
                    dist_now      = dist_mid
            
            if atm_id_belong != i:
                IP_cluster[min(i,atm_id_belong), max(i,atm_id_belong)].append(ip_id)
            else:
                IP_cluster[i].append(ip_id)
    
    return IP_cluster, bond_mid_coord

C = 7
M = 5
CUTOFF = 1e-8

if __name__ == "__main__":
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    KE_CUTOFF = 32
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
    
    Ls = [1, 2, 2]
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell = build_supercell(atm, prim_a, Ls=Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    
    ConnectionInfo = {
        "C":{
            "C":4
        }
    }
    
    # exit(1)
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)
    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    
    atm_coords = []
    for i in range(cell.natm):
        print('%s %s  charge %f  xyz %s' % (cell.atom_symbol(i),
                                            cell.atom_pure_symbol(i),
                                            cell.atom_charge(i),
                                            cell.atom_coord(i)))
        atm_coords.append(cell.atom_coord(i))

    print("Atoms' charges in a vector\n%s" % cell.atom_charges())
    print("Atoms' coordinates in an array\n%s" % cell.atom_coords())
    
    atm_coords = np.array(atm_coords)
    
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
    
    # print("coords = ", coords)
    # print(np.max(coords[:,0]), np.max(coords[:,1]), np.max(coords[:,2]))
    # print(np.min(coords[:,0]), np.min(coords[:,1]), np.min(coords[:,2]))
    # print("atm_coords = ", atm_coords)
    # exit(1)
    # print("pbc_isdf_info.coords = ", pbc_isdf_info.coords)
    IP_cluster, bond_mid = _cluster_IP(cell, ConnectionInfo, pbc_isdf_info.partition, pbc_isdf_info.IP_ID, coords)

    # print("IP_cluster = ", IP_cluster)

    # lattice_vectors = cell.lattice_vectors()
    # for key in IP_cluster:
    #     coords_cluster = coords[IP_cluster[key]]
    #     for coord_ in coords_cluster:
    #         if coord_[0] < 0:
    #             coord_[0] += lattice_vectors[0][0]
    #         if coord_[1] < 0:
    #             coord_[1] += lattice_vectors[1][1]
    #         if coord_[2] < 0:
    #             coord_[2] += lattice_vectors[2][2]
    #     print("key = ", key)
    #     # print("coords_cluster = ", coords_cluster)
    #     if isinstance(key, int):
    #         key = (key,)
    #     atm_involved = list(key)
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(coords_cluster[:,0], coords_cluster[:,1], coords_cluster[:,2], s=50)
    #     ax.scatter(coords_cluster[:,0]+lattice_vectors[0][0], coords_cluster[:,1]+lattice_vectors[1][1], coords_cluster[:,2], s=50)
    #     ax.scatter(coords_cluster[:,0]+lattice_vectors[0][0], coords_cluster[:,1]+lattice_vectors[1][1], coords_cluster[:,2]+lattice_vectors[2][2], s=50)
    #     ax.scatter(coords_cluster[:,0]+lattice_vectors[0][0], coords_cluster[:,1], coords_cluster[:,2]+lattice_vectors[2][2], s=50)
    #     ax.scatter(coords_cluster[:,0]+lattice_vectors[0][0], coords_cluster[:,1], coords_cluster[:,2], s=50)
    #     ax.scatter(coords_cluster[:,0], coords_cluster[:,1]+lattice_vectors[1][1], coords_cluster[:,2], s=50)
    #     ax.scatter(coords_cluster[:,0], coords_cluster[:,1], coords_cluster[:,2]+lattice_vectors[2][2], s=50)
    #     ax.scatter(coords_cluster[:,0], coords_cluster[:,1]+lattice_vectors[1][1], coords_cluster[:,2]+lattice_vectors[2][2], s=50)
    #     ## print atm_coords ##
    #     # ax.scatter(atm_coords[:,0], atm_coords[:,1], atm_coords[:,2], c='r', marker='o', s=100)
    #     natm = cell.natm
    #     if len(atm_involved) == 2:
    #         coord_mid = bond_mid[(min(atm_involved), max(atm_involved))]
    #         ax.scatter(coord_mid[0], coord_mid[1], coord_mid[2], c='g', marker='o', s=150)
    #     for i in range(natm):
    #         if i in atm_involved:
    #             ax.scatter(atm_coords[i,0], atm_coords[i,1], atm_coords[i,2], c='r', marker='o', s=100)
    #         else:
    #             ax.scatter(atm_coords[i,0], atm_coords[i,1], atm_coords[i,2], c='b', marker='o', s=100)
    #     plt.show()

    # exit(1)
    
    # V_R = pbc_isdf_info.V_R
    aux_bas = pbc_isdf_info.aux_basis
    
    ##### check the sparsity of A and B ##### 
    
    A, B = pbc_isdf_info.get_A_B()
    
    e, h = np.linalg.eigh(A)
    
    print("e = ", e)
    size = 0
    for i in range(B.shape[0]):
        tmp = B[i:i+1, :].reshape(-1, *mesh)
        # print("tmp.shape = ")
        tmp_HOSVD = HOSVD.HOSVD(tmp, cutoff=1e-20, rela_cutoff=1e-10)
        # print(tmp_HOSVD.S[1])
        # print("row i ", i, " tmp_HOSVD.Bshape = ", tmp_HOSVD.Bshape)
        size += tmp_HOSVD.size()
        tmp_full = tmp_HOSVD.getFullMat()
        B[i:i+1, :] = tmp_full.reshape(1, -1)
    print("compress = %15.8f"%(size/B.size))
    
    aux_basis = np.dot(h.T, B)
    aux_basis = (1.0/e).reshape(-1,1) * aux_basis
    aux_basis = np.dot(h, aux_basis)
    
    diff = aux_bas - aux_basis 
    
    print("Diff = %15.8e" % (np.linalg.norm(diff)))
        
    # exit(1)
        
    # pbc_isdf_info.aux_basis = aux_basis
    # aux_bas = None
    # pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)
    
    # exit(1)
    
    # tmp = aux_bas[0:1, :].reshape(-1, *mesh)
    # tmp_HOSVD = HOSVD.HOSVD(tmp)
    
    # perform the compression row by row 
    
    # for i in range(aux_bas.shape[0]):
    #     tmp = V_R[i:i+1, :].reshape(-1, *mesh)
    #     tmp_HOSVD = HOSVD.HOSVD(tmp, cutoff=1e-6)
    #     print("row i ", i, " tmp_HOSVD.Bshape = ", tmp_HOSVD.Bshape)
    #     # print(tmp_HOSVD.S)
    #     tmp_full = tmp_HOSVD.getFullMat()
    #     V_R[i:i+1, :] = tmp_full.reshape(1, -1)
    # pbc_isdf_info.V_R = V_R
    
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
    
    # exit(1)
    
    dm1 = mf.make_rdm1()
    
    aoR = pbc_isdf_info.aoR
    aoRg = pbc_isdf_info.aoRg
    
    ###################### individual compression ######################
    
    V_R = pbc_isdf_info.V_R
    
    size = 0
    for i in range(aux_bas.shape[0]):
        tmp = aux_bas[i:i+1, :].reshape(-1, *mesh)
        tmp_HOSVD = HOSVD.HOSVD(tmp, cutoff=CUTOFF*10)
        # print("row i ", i, " tmp_HOSVD.Bshape = ", tmp_HOSVD.Bshape)
        size += tmp_HOSVD.size()
    
    print("compression of aux_bas = %15.8f" % (size / aux_bas.size))
    print("mesh = ", mesh)
    
    # exit(1)
    
    DM_RgR = lib.ddot(aoRg.T, dm1)
    DM_RgR = lib.ddot(DM_RgR, aoR)
    
    size = 0
    for i in range(aux_bas.shape[0]):
        tmp = DM_RgR[i:i+1, :].reshape(-1, *mesh)
        tmp_HOSVD = HOSVD.HOSVD(tmp, cutoff=CUTOFF)
        # print("row i ", i, " DM_tmp_HOSVD.Bshape = ", tmp_HOSVD.Bshape)
        size += tmp_HOSVD.size()
    
    print("compression of DM_RgR = %15.8f" % (size / DM_RgR.size))
    print("mesh = ", mesh)
    
    # extract W_R * DM_RgR # 
    
    DM_RgR = DM_RgR * V_R
    
    size = 0
    for i in range(aux_bas.shape[0]):
        tmp = DM_RgR[i:i+1, :].reshape(-1, *mesh)
        tmp_HOSVD = HOSVD.HOSVD(tmp, cutoff=CUTOFF)
        # print("row i ", i, " DM_tmp_HOSVD.Bshape = ", tmp_HOSVD.Bshape)
        size += tmp_HOSVD.size()
    
    print("compression of DM_RgR * V_R = %15.8f" % (size / DM_RgR.size))
    print("mesh = ", mesh)
    
    DM_R = lib.ddot(dm1, aoR)
    DM_R = DM_R * aoR 
    DM_R = np.einsum("ip->p", DM_R)
    DM_R = DM_R.reshape(-1, *mesh)
    tmp_HOSVD = HOSVD.HOSVD(DM_R, cutoff=CUTOFF)
    print("DM_R_HOSVD.Bshape = ", tmp_HOSVD.Bshape) 
    
    DM_RgR = None
    DM_R = None
    
    ###################### cluster compression ######################
    
    flat_IP_ID = {}
    
    for id_, ip_id in enumerate(pbc_isdf_info.IP_ID):
        flat_IP_ID[ip_id] = id_
        
    for key in IP_cluster:
        IP_cluster_now = IP_cluster[key]
        flat_id = [flat_IP_ID[ip_id] for ip_id in IP_cluster_now]
        flat_id = np.array(flat_id)
        
        aux_bas_now = aux_bas[flat_id, :]
        print("aux_bas_now.shape = ", aux_bas_now.shape)
        print("mesh = ", mesh)
        aux_bas_now = aux_bas_now.reshape(-1, *mesh)
        tmp_HOSVD = HOSVD.HOSVD(aux_bas_now, cutoff=CUTOFF)
        size = len(IP_cluster_now) * np.prod(mesh)
        print("Bshape of tmp_HOSVD = ", tmp_HOSVD.Bshape)
        size_compressed = tmp_HOSVD.size()
        print("compression of aux_bas of cluster %s = %15.8f" % (key, size_compressed / size))
    
    DM_RgR = lib.ddot(aoRg.T, dm1)
    DM_RgR = lib.ddot(DM_RgR, aoR)
    
    for key in IP_cluster:
        IP_cluster_now = IP_cluster[key]
        flat_id = [flat_IP_ID[ip_id] for ip_id in IP_cluster_now]
        flat_id = np.array(flat_id)
        
        DM_RgR_now = DM_RgR[flat_id, :]
        print("DM_RgR_now.shape = ", DM_RgR_now.shape)
        print("mesh = ", mesh)
        DM_RgR_now = DM_RgR_now.reshape(-1, *mesh)
        tmp_HOSVD = HOSVD.HOSVD(DM_RgR_now, cutoff=CUTOFF)
        size = len(IP_cluster_now) * np.prod(mesh)
        print("Bshape of tmp_HOSVD = ", tmp_HOSVD.Bshape)
        size_compressed = tmp_HOSVD.size()
        print("compression of DM_RgR of cluster %s = %15.8f" % (key, size_compressed / size))
        
    DM_RgR = DM_RgR * V_R
    
    for key in IP_cluster:
        IP_cluster_now = IP_cluster[key]
        flat_id = [flat_IP_ID[ip_id] for ip_id in IP_cluster_now]
        flat_id = np.array(flat_id)
        
        DM_RgR_now = DM_RgR[flat_id, :]
        print("DM_RgR_now.shape = ", DM_RgR_now.shape)
        print("mesh = ", mesh)
        DM_RgR_now = DM_RgR_now.reshape(-1, *mesh)
        tmp_HOSVD = HOSVD.HOSVD(DM_RgR_now, cutoff=CUTOFF)
        size = len(IP_cluster_now) * np.prod(mesh)
        print("Bshape of tmp_HOSVD = ", tmp_HOSVD.Bshape)
        size_compressed = tmp_HOSVD.size()
        print("compression of DM_RgR * V_R of cluster %s = %15.8f" % (key, size_compressed / size)) # we should not use the cluster compression here ! 