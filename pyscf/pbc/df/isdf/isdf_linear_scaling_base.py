#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

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

from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info

import pyscf.pbc.df.isdf.isdf_outcore as ISDF_outcore
import pyscf.pbc.df.isdf.isdf_fast as ISDF
import pyscf.pbc.df.isdf.isdf_split_grid as ISDF_split_grid
import pyscf.pbc.df.isdf.isdf_k as ISDF_K

from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch

from pyscf.pbc.df.isdf.isdf_fast_mpi import get_jk_dm_mpi

import ctypes, sys

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto


############ build atm connection graph ############

class AtmConnectionInfo:
    def __init__(self, cell:Cell, atmID, distance_matrix, precision, rcut, rcut_max, atm_to_bas):
        '''
        rcut: the cutoff radius of each bas
        '''
        
        self.precision = precision
        self.atmID = atmID
        self.atmID_connection = np.where(distance_matrix[atmID] < rcut_max)[0]
        self.distance = distance_matrix[atmID][self.atmID_connection]
        self.atm_connected_info = list(zip(self.atmID_connection, self.distance))
        # sort by distance 
        self.atm_connected_info.sort(key=lambda x: x[1])
        self.bas_range = np.arange(atm_to_bas[atmID][0], atm_to_bas[atmID][1])
        self.bas_cut = rcut[atm_to_bas[atmID][0]:atm_to_bas[atmID][1]]
    
    def __repr__(self):
        return "atmID = %d, atm_connected_info = %s, bas_range = %s, bas_cut = %s" % (self.atmID, self.atm_connected_info, self.bas_range, self.bas_cut)

class aoR_Holder:
    def __init__(self, aoR, ao_involved, local_gridID_begin, local_gridID_end, global_gridID_begin, global_gridID_end):
        
        assert aoR.shape[0] == len(ao_involved)
        assert aoR.shape[1] == local_gridID_end - local_gridID_begin
        assert aoR.shape[1] == global_gridID_end - global_gridID_begin
        
        self.aoR = aoR
        self.ao_involved = np.array(ao_involved, dtype=np.int32)
        self.local_gridID_begin = local_gridID_begin
        self.local_gridID_end = local_gridID_end
        self.global_gridID_begin = global_gridID_begin
        self.global_gridID_end = global_gridID_end
    
    def size(self):
        return self.aoR.nbytes + self.ao_involved.nbytes

    def todense(self, nao):
        aoR = np.zeros((nao, self.aoR.shape[1]))
        aoR[self.ao_involved] = self.aoR
        return aoR

def _HadamardProduct(a:aoR_Holder, b:aoR_Holder, InPlaceB=True):
    if InPlaceB:
        res = b
    else:
        res = a
    
    assert a.aoR.shape == b.aoR.shape
    assert a.local_gridID_begin == b.local_gridID_begin
    assert a.global_gridID_begin == b.global_gridID_begin
    assert a.local_gridID_end == b.local_gridID_end
    assert a.global_gridID_end == b.global_gridID_end
    
    lib.cwise_mul(a.aoR, b.aoR, res.aoR)
    
    return res

def _get_aoR_holders_memory(aoR_holders:list[aoR_Holder]):
    return sum([_aoR_holder.size() for _aoR_holder in aoR_holders])

def _pack_aoR_holder(aoR_holders:list[aoR_Holder], nao):
    has_involved = [False] * nao
    
    nGrid = 0
    for _aoR_holder in aoR_holders:
        for i in _aoR_holder.ao_involved:
            has_involved[i] = True  
        nGrid += _aoR_holder.aoR.shape[1]
    
    ao2loc = [-1] * nao 
    loc_now = 0
    for ao_id, involved in enumerate(has_involved):
        if involved:
            ao2loc[ao_id] = loc_now
            loc_now += 1
    nao_involved = loc_now  
    
    aoR_packed = np.zeros((nao_involved, nGrid))
    
    fn_pack = getattr(libpbc, "_Pack_Matrix_SparseRow_DenseCol", None)
    assert fn_pack is not None
    
    
    grid_begin_id = 0
    for _aoR_holder in aoR_holders:
        loc_packed = np.zeros((_aoR_holder.aoR.shape[0]), dtype=np.int32)
        grid_end_id = grid_begin_id + _aoR_holder.aoR.shape[1]
        for loc, ao_id in enumerate(_aoR_holder.ao_involved):
            loc_packed[loc] = ao2loc[ao_id]
        # print("loc_packed = ", loc_packed)
        # print("grid_begin_id = %d, grid_end_id = %d" % (grid_begin_id, grid_end_id))
        # aoR_packed[loc_packed, grid_begin_id:grid_end_id] = _aoR_holder.aoR
        fn_pack(
            aoR_packed.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(aoR_packed.shape[0]),
            ctypes.c_int(aoR_packed.shape[1]),
            _aoR_holder.aoR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(_aoR_holder.aoR.shape[0]),
            ctypes.c_int(_aoR_holder.aoR.shape[1]),
            loc_packed.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(grid_begin_id),
            ctypes.c_int(grid_end_id)
        )
        grid_begin_id = grid_end_id
    ao_packed_invovled = np.array([i for i in range(nao) if has_involved[i]], dtype=np.int32)

    assert nGrid == grid_begin_id
    local_gridID_begin = 0
    local_gridID_end = nGrid
    global_gridID_begin = 0
    global_gridID_end = nGrid
    
    return aoR_Holder(aoR_packed, ao_packed_invovled, local_gridID_begin, local_gridID_end, global_gridID_begin, global_gridID_end)

# get the rcut #

def _atm_to_bas(cell:Cell):
    shl_atm = []
        
    natm = cell.natm
        
    for i in range(natm):
        shl_atm.append([None, None])
        
    for i in range(cell.nbas):
        atm_id = cell.bas_atom(i)
        if shl_atm[atm_id][0] is None:
            shl_atm[atm_id][0] = i
        shl_atm[atm_id][1] = i+1
    
    return shl_atm

def _estimate_rcut(cell, ngrids, precision):
    '''Cutoff raidus, above which each shell decays to a value less than the
    required precsion'''
    weight = numpy.sqrt(cell.vol/ngrids) # note the weight ! 
    log_prec = numpy.log(precision/weight)
    rcut = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        r[r < 1.] = 1.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        rcut.append(r.max())
    return numpy.array(rcut)

# the distance graph # 

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

def get_cell_distance_matrix(cell:Cell):
    '''
    get the distance matrix of the cell
    '''
    a = cell.lattice_vectors()
    n = cell.natm
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i][j] = _distance_translation(cell.atom_coord(i), cell.atom_coord(j), a)
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix

############ algorithm based on the distance graph and AtmConnectionInfo ############

def get_partition(cell:Cell, coords, AtmConnectionInfoList:list[AtmConnectionInfo], 
                  Ls=[3,3,3]): # by default split the cell into 4x4x4 supercell
    
    ##### construct the box info #####

    mesh = cell.mesh
    lattice_vector = cell.lattice_vectors()
    
    mesh_box = np.array([0,0,0])
    nbox = np.array([0,0,0])
    if mesh[0] % Ls[0] != 0:
        mesh_box[0] = mesh[0] // Ls[0] + 1
        nbox[0] = mesh[0] // mesh_box[0] + 1
    else:
        mesh_box[0] = mesh[0] // Ls[0]
        nbox[0] = mesh[0] // mesh_box[0]
    if mesh[1] % Ls[1] != 0:
        mesh_box[1] = mesh[1] // Ls[1] + 1
        nbox[1] = mesh[1] // mesh_box[1] + 1
    else:
        mesh_box[1] = mesh[1] // Ls[1]
        nbox[1] = mesh[1] // mesh_box[1]
    if mesh[2] % Ls[2] != 0:
        mesh_box[2] = mesh[2] // Ls[2] + 1
        nbox[2] = mesh[2] // mesh_box[2] + 1
    else:
        mesh_box[2] = mesh[2] // Ls[2]
        nbox[2] = mesh[2] // mesh_box[2]
    
    print("mesh = ", mesh)
    print("mesh_box = ", mesh_box)
    print("nbox = ", nbox)
    # print("lattice_vector = ", lattice_vector)
    Ls_box = [lattice_vector[0] / mesh[0] * mesh_box[0], lattice_vector[1] / mesh[1] * mesh_box[1], lattice_vector[2] / mesh[2] * mesh_box[2]]
    print("Ls_box = ", Ls_box)
    assert Ls_box[0][0] < 3.0
    assert Ls_box[1][1] < 3.0
    assert Ls_box[2][2] < 3.0 # the box cannot be too large
    
    ##### helper functions ##### 
    
    def get_box_id(x, y, z):
        # print("x = %f, y = %f, z = %f" % (x, y, z))
        # print(Ls_box)
        ix = int(x // Ls_box[0][0])
        iy = int(y // Ls_box[1][1])
        iz = int(z // Ls_box[2][2])
        return (ix, iy, iz)
        
    def get_box_id_from_coord(coord):
        return get_box_id(coord[0], coord[1], coord[2])

    def get_mesh_id(ix, iy, iz):
        return ix * mesh[1] * mesh[2] + iy * mesh[2] + iz

    ##### build info between atm and box id #####

    atm_box_id = []
    box_2_atm = {}

    for i in range(cell.natm):
        # print(cell.atom_coord(i))
        print("atm %d at %s is in box %s" % (i, cell.atom_coord(i), get_box_id_from_coord(cell.atom_coord(i))))
        box_id = get_box_id_from_coord(cell.atom_coord(i))
        atm_box_id.append(box_id)
        if box_id not in box_2_atm:
            box_2_atm[box_id] = [i]
        else:
            box_2_atm[box_id].append(i)
    
    print("atm_box_id = ", atm_box_id)
    print("box_2_atm = ", box_2_atm)

    ######## a rough partition of the cell based on distance only ######## 

    partition_rough = []
    for i in range(cell.natm):
        partition_rough.append([])

    for ix in range(nbox[0]):
        for iy in range(nbox[1]):
            for iz in range(nbox[2]):
                box_id = (ix, iy, iz)
                
                #### construct the grid ID ####
                
                mesh_x_begin = min(ix * mesh_box[0], mesh[0])
                mesh_x_end = min((ix+1) * mesh_box[0], mesh[0])

                if mesh_x_begin == mesh_x_end:
                    continue
            
                mesh_y_begin = min(iy * mesh_box[1], mesh[1])
                mesh_y_end = min((iy+1) * mesh_box[1], mesh[1])
                
                if mesh_y_begin == mesh_y_end:
                    continue
                
                mesh_z_begin = min(iz * mesh_box[2], mesh[2])
                mesh_z_end = min((iz+1) * mesh_box[2], mesh[2])
                
                if mesh_z_begin == mesh_z_end:
                    continue
            
                grid_ID = []
                
                for i in range(mesh_x_begin, mesh_x_end):
                    for j in range(mesh_y_begin, mesh_y_end):
                        for k in range(mesh_z_begin, mesh_z_end):
                            grid_ID.append(get_mesh_id(i, j, k))
                
                grid_ID = np.array(grid_ID, dtype=np.int32)
                
                if box_id in box_2_atm:
                    partition_rough[box_2_atm[box_id][0]].extend(grid_ID)
                else:
                    # random pickup one coord in the box 
                    grid_ID_random_pick = grid_ID[np.random.randint(0, len(grid_ID))]
                    grid_coord = coords[grid_ID_random_pick]
                    atm_id = None
                    min_distance = 1e10
                    for i in range(cell.natm):
                        distance_now = _distance_translation(grid_coord, cell.atom_coord(i), lattice_vector)
                        if distance_now < min_distance:
                            min_distance = distance_now
                            atm_id = i
                    partition_rough[atm_id].extend(grid_ID)
        
    # print("partition_rough = ", partition_rough)
    
    len_grid_involved = 0
    for atm_id, x in enumerate(partition_rough):
        print("atm %d involved %d grids" % (atm_id, len(x)))
        len_grid_involved += len(x)
    assert len_grid_involved == mesh[0] * mesh[1] * mesh[2]
    
    ######## refine the partition based on the AtmConnectionInfo ########
    
    partition = []
    for i in range(cell.natm):
        partition.append([])
    
    DISTANCE_CUTOFF = 8 # suitable for cuprates ! 
    
    ao_loc = cell.ao_loc_nr()
    print("nao_intot = ", ao_loc[-1])
    for atm_id in range(cell.natm):
        atm_involved = []
        
        ## pick up atms with distance < DISTANCE_CUTOFF ##
        
        for atm_id_other, distance in AtmConnectionInfoList[atm_id].atm_connected_info:
            if distance < DISTANCE_CUTOFF:
                atm_involved.append(atm_id_other) 
        atm_involved.sort()
        print("atm %d involved atm = %s" % (atm_id, atm_involved))
        
        ## get the involved ao ##
        
        nao_invovled = 0
        for atm_id_other in atm_involved:
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end = AtmConnectionInfoList[atm_id_other].bas_range[-1]+1
            nao_invovled += ao_loc[shl_end] - ao_loc[shl_begin]
        print("atm %d involved %d ao" % (atm_id, nao_invovled))
        
        grid_ID = partition_rough[atm_id]
        
        ## determine the partition more clever by aoR ##
        
        aoR = np.zeros((nao_invovled, len(grid_ID)))
        ao_loc_now = 0
        bas_2_atm_ID = []
        for atm_id_other in atm_involved:
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end = AtmConnectionInfoList[atm_id_other].bas_range[-1]+1
            aoR_tmp = ISDF_eval_gto(cell, coords= coords[grid_ID], shls_slice=(shl_begin, shl_end))
            for _ in range(aoR_tmp.shape[0]):
                bas_2_atm_ID.append(atm_id_other)
            aoR[ao_loc_now:ao_loc_now+aoR_tmp.shape[0]] = aoR_tmp
            ao_loc_now += aoR_tmp.shape[0]
            aoR_tmp = None
        assert ao_loc_now == nao_invovled
        
        partition_tmp = np.argmax(np.abs(aoR), axis=0)
        partition_tmp = np.asarray([bas_2_atm_ID[x] for x in partition_tmp])
        for grid_id, atm_id in zip(grid_ID, partition_tmp):
            partition[atm_id].append(grid_id)
    
    len_grid_involved = 0
    for atm_id, x in enumerate(partition):
        print("atm %d involved %d grids" % (atm_id, len(x)))
        len_grid_involved += len(x)
    assert len_grid_involved == mesh[0] * mesh[1] * mesh[2]
    
    del partition_rough
    del aoR_tmp
    del aoR
    
    return partition

def get_aoR(cell:Cell, coords, partition, AtmConnectionInfoList:list[AtmConnectionInfo], distributed = False, use_mpi=False):
    
    weight = np.sqrt(cell.vol / coords.shape[0])
    
    if use_mpi:
        raise NotImplementedError("not implemented yet")
    
    RcutMax = -1e10
    
    for _info_ in AtmConnectionInfoList:
        RcutMax = max(RcutMax, np.max(_info_.bas_cut))
    
    precision = AtmConnectionInfoList[0].precision
        
    aoR_holder = []
    
    for _ in range(cell.natm):
        aoR_holder.append(None)
    
    local_gridID_begin = 0
    global_gridID_begin = 0
    ao_loc = cell.ao_loc_nr()
    
    for atm_id, grid_ID in enumerate(partition):
        
        ##### find the involved atms within RcutMax #####
        
        atm_involved = []
        for atm_id_other, distance in AtmConnectionInfoList[atm_id].atm_connected_info:
            if distance < RcutMax:
                atm_involved.append(atm_id_other)
        atm_involved.sort()
        
        ##### get the involved ao #####
        
        nao_invovled = 0
        for atm_id_other in atm_involved:
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end = AtmConnectionInfoList[atm_id_other].bas_range[-1]+1
            nao_invovled += ao_loc[shl_end] - ao_loc[shl_begin]
        
        # print("atm %d involved %d ao" % (atm_id, nao_invovled))
        
        aoR = np.zeros((nao_invovled, len(grid_ID)))
        bas_id = []

        ao_loc_now = 0
        
        for atm_id_other in atm_involved:
            
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end = AtmConnectionInfoList[atm_id_other].bas_range[-1]+1
            bas_id.extend(np.arange(ao_loc[shl_begin], ao_loc[shl_end]))
            
            nshell = len(AtmConnectionInfoList[atm_id_other].bas_range)
            distance = distance_matrix[atm_id, atm_id_other]
            rcut = AtmConnectionInfoList[atm_id_other].bas_cut
            
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end   = AtmConnectionInfoList[atm_id_other].bas_range[0]
            ao_loc_begin = ao_loc_now
            ao_loc_end = ao_loc_now
            
            for shl_now, shell_rcut in zip(AtmConnectionInfoList[atm_id_other].bas_range, rcut):
                shl_end = shl_now+1
                ao_loc_end += ao_loc[shl_now+1] - ao_loc[shl_now]
                if shell_rcut < distance:
                    # print("shl_now = %d, shl_end = %d, ao_loc_begin = %d, ao_loc_end = %d" % (shl_now, shl_end, ao_loc_begin, ao_loc_end))
                    # print("shell_rcut = %f, distance = %f" % (shell_rcut, distance))
                    ### calcualte the aoR for this shell ###
                    if shl_begin < shl_end:
                        aoR_tmp = ISDF_eval_gto(cell, coords=coords[grid_ID], shls_slice=(shl_begin, shl_end)) * weight
                        aoR[ao_loc_begin:ao_loc_end] = aoR_tmp
                    ao_loc_begin = ao_loc_end
                    ao_loc_end = ao_loc_begin
                    ao_loc_now = ao_loc_end
                    shl_end = shl_now+1
                    shl_begin = shl_end
                    # print("after calculation, ao_loc_now = %d" % ao_loc_now)
            
            ### the final calculation for the last shell ###
            
            if shl_begin < shl_end:
                aoR_tmp = ISDF_eval_gto(cell, coords=coords[grid_ID], shls_slice=(shl_begin, shl_end)) * weight
                aoR[ao_loc_begin:ao_loc_end] = aoR_tmp
            
            ao_loc_now = ao_loc_end
        
        assert ao_loc_now == nao_invovled
        
        ##### screening the aoR ##### 
        
        max_row = np.max(np.abs(aoR), axis=1)
        where = np.where(max_row > precision)[0]
        aoR = aoR[where]
        bas_id = np.array(bas_id)[where]
        
        aoR_holder[atm_id] = aoR_Holder(aoR, bas_id, local_gridID_begin, local_gridID_begin+len(grid_ID), global_gridID_begin, global_gridID_begin+len(grid_ID))
        
        local_gridID_begin += len(grid_ID)
        global_gridID_begin += len(grid_ID)
                        
    del aoR
    del aoR_tmp
    
    return aoR_holder

############ get estimate on the strcuture of Density Matrix ############



############ get calculation scheme for V W and rhoG and DMg ############


from pyscf.lib.parameters import BOHR

TARGET_PRECISION = 1e-9

if __name__ == '__main__':
    
    prim_a = np.array(
                    [[14.572056092/2, 0.000000000, 0.000000000],
                     [0.000000000, 14.572056092/2, 0.000000000],
                     [0.000000000, 0.000000000,  6.010273939],]) * BOHR
    atm = [
['Cu1',	(1.927800,	1.927800,	1.590250)],
['O1',	(1.927800,	0.000000,	1.590250)],
['O1',	(0.000000,	1.927800,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
    ]
    
    basis = {
        'Cu1':'ecpccpvdz', 'Cu2':'ecpccpvdz', 'O1': 'ecpccpvdz', 'Ca':'ecpccpvdz'
    }
    
    pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
    
    ke_cutoff = 256 
    
    prim_cell = ISDF_K.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    
    supercell = [2, 2, 1]
    # supercell = [4,4,2]
    
    mesh = [supercell[0] * prim_mesh[0], supercell[1] * prim_mesh[1], supercell[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell = ISDF_K.build_supercell(atm, prim_a, Ls = supercell, ke_cutoff=ke_cutoff, mesh=mesh, basis=basis, pseudo=pseudo)
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp  = MultiGridFFTDF2(cell)
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    assert coords is not None
    
    distance_matrix = get_cell_distance_matrix(cell)
    
    # for i in range(cell.natm):
    #     for j in range(cell.natm):
    #         print("distance between atom %d and atom %d is %f" % (i, j, distance_matrix[i][j]))
    
    weight = np.sqrt(cell.vol / coords.shape[0])
    
    precision = TARGET_PRECISION
    rcut = _estimate_rcut(cell, coords.shape[0], precision)
    rcut_max = np.max(rcut)
    
    # import matplotlib.pyplot as plt
    
    # plt.hist(rcut, bins=100)
    # plt.xscale('log')
    # plt.show()
    
    # print("rcut = ", rcut)
    # print("precision = ", precision)
    # print("max_rcut = ", np.max(rcut))
    # print("nbas = ", cell.nbas)
    # print("number of rcut < 5 = ", np.sum(rcut < 5))
    # print("number of rcut < 15 = ", np.sum(rcut < 15))
    # print("number of rcut > 25 = ", np.sum(rcut > 25))
    # print("number of atm pair < 5 = ", np.sum(distance_matrix < 5))
    # print("number of atm pair < 15 = ", np.sum(distance_matrix < 15))
    # print("number of atm pair > rcut_max = ", np.sum(distance_matrix > rcut_max))
    
    atm_2_bas = _atm_to_bas(cell)
    # print("atm_2_bas = ", atm_2_bas)
    # exit(1)
    AtmConnectionInfoList = []
    
    
    for i in range(cell.natm):
        tmp = AtmConnectionInfo(cell, i, distance_matrix, precision, rcut, rcut_max, atm_2_bas)
        print(tmp)
        AtmConnectionInfoList.append(tmp)
        
        # test seraialization used in MPI # 
        
        from mpi4py import MPI
        
        # tmp_serialize = MPI.pickle.dumps(tmp)
        # print("tmp_serialize = ", tmp_serialize)
        # print(tmp_serialize.__class__)
        # tmp2 = MPI.pickle.loads(tmp_serialize)
        # print(tmp2)
        # deserialized_bytes = np.frombuffer(tmp_serialize, dtype=np.uint8)
        # print(deserialized_bytes)
        # tmp_serialize2 = deserialized_bytes.tobytes()
        # tmp3 = MPI.pickle.loads(tmp_serialize2)
        # print(tmp3)
    AtmConnectionInfoList = []
    
    # precision = TARGET_PRECISION / weight
    precision = TARGET_PRECISION
    rcut = _estimate_rcut(cell, coords.shape[0], precision)
    rcut_max = np.max(rcut)
    for i in range(cell.natm):
        tmp = AtmConnectionInfo(cell, i, distance_matrix, precision, rcut, rcut_max, atm_2_bas)
        print(tmp)
        AtmConnectionInfoList.append(tmp)
    
    partition = get_partition(cell, coords, AtmConnectionInfoList, Ls=[supercell[0]*3, supercell[1]*3, supercell[2]*3])
    
    aoR_list = get_aoR(cell, coords, partition, AtmConnectionInfoList)
    
    print("memory of aoR_list = ", _get_aoR_holders_memory(aoR_list))
    for x in aoR_list:
        print("nao_involved = ", x.aoR.shape[0])
    
    grid_reordered = []
    for _partition_ in partition:
        grid_reordered.extend(_partition_)
    grid_reordered = np.array(grid_reordered, dtype=np.int32)
    
    ### check the partition ### 
    
    for _id_, _partition_ in enumerate(partition):
        aoR_benchmark = ISDF_eval_gto(cell, coords=coords[_partition_]) * weight
        max_row = np.max(np.abs(aoR_benchmark), axis=1)
        where = np.where(max_row > precision)[0]
        # aoR_benchmark = aoR_benchmark[where]
        aoR_packed = aoR_list[_id_].todense(cell.nao_nr())
        diff = np.max(np.abs(aoR_packed - aoR_benchmark))
        print("atm %d, diff = %.4e" % (_id_, diff))
        aoR_benchmark = aoR_benchmark[where]
        print("atm %d, aoR shape = %s, aoR_benchmark shape = %s" % (_id_, aoR_list[_id_].aoR.shape, aoR_benchmark.shape))
    
    ### check the pack ###
    
    aoR = _pack_aoR_holder(aoR_list, cell.nao_nr())
    aoR_benchmark =  ISDF_eval_gto(cell, coords=coords[grid_reordered]) * weight
    # print("aoR.aoR", aoR.aoR[0,0])
    # print("aoR_benchmark", aoR_benchmark[0,0])
    diff = np.max(np.abs(aoR.aoR - aoR_benchmark))
    print("diff = %.4e" % diff)