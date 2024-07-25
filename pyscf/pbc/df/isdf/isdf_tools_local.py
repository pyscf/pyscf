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

########## pyscf module ##########

import copy
from functools import reduce
import numpy as np
import pyscf
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *
import pyscf.pbc.df.ft_ao as ft_ao
from pyscf.pbc.df import aft, rsdf_builder, aft_jk

########## isdf  module ##########

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_ao2mo as isdf_ao2mo
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

########## sys   module ##########

import ctypes, sys
from multiprocessing import Pool
libpbc = lib.load_library('libpbc')

########## global parameter ##########

DISTANCE_CUTOFF = 16 # suitable for cuprates ! 

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
        '''
        currently local_gridID_begin, local_gridID_end is not useful
        '''
        
        assert aoR.shape[0] == len(ao_involved)
        assert (local_gridID_end - local_gridID_begin) == (global_gridID_end - global_gridID_begin)
        assert aoR.shape[1] <= (global_gridID_end - global_gridID_begin)
        # assert aoR.shape[1] == local_gridID_end - local_gridID_begin
        # assert aoR.shape[1] == global_gridID_end - global_gridID_begin
        # if aoR.shape[1] != (global_gridID_end - global_gridID_begin):
        self.ngrid_tot  = global_gridID_end - global_gridID_begin
        self.ngrid_kept = aoR.shape[1]
        
        self.aoR = aoR
        self.ao_involved = np.array(ao_involved, dtype=np.int32)
        self.nao_involved = len(ao_involved)
        self.local_gridID_begin = local_gridID_begin
        self.local_gridID_end = local_gridID_end
        self.global_gridID_begin = global_gridID_begin
        self.global_gridID_end = global_gridID_end
        self.nCompact = self.nao_involved  ## by default all orbitals are compact
        
        ## build ao_involved segment ## 
        
        self.ao_involved_sorted = np.sort(self.ao_involved)
        self.aoR         = self.aoR[np.argsort(self.ao_involved)]
        self.ao_involved = self.ao_involved_sorted 
        
        diff            = np.diff(self.ao_involved)
        segment_indices = np.where(diff > 1)[0] + 1
        segments        = np.split(self.ao_involved, segment_indices)
        
        self.segments = []
        if len(segments) == 1 and len(segments[0]) == 0:
            self.segments.append(0)
        else:
            loc_begin = 0
            for segment in segments:
                self.segments.append(loc_begin)
                self.segments.append(segment[0])
                self.segments.append(segment[-1]+1)
                loc_begin += len(segment)
            self.segments.append(loc_begin)
        
        self.segments = np.array(self.segments, dtype=np.int32)

        segments = None
    
    def RangeSeparation(self, IsCompact:np.ndarray):
        ordering_C = []
        ordering_D = []
        nao_involved = len(self.ao_involved)
        for i in range(nao_involved):
            if IsCompact[self.ao_involved[i]]:
                ordering_C.append(i)
            else:
                ordering_D.append(i)
        self.nCompact = len(ordering_C)
        ordering = ordering_C
        ordering.extend(ordering_D)
        ordering = np.array(ordering, dtype=np.int32)
        self.aoR = self.aoR[ordering].copy()
        self.ao_involved = self.ao_involved[ordering].copy()
        # print("ordering = ", ordering)
        # print("nCompact = ", self.nCompact)
        for i in range(self.nCompact):
            assert IsCompact[self.ao_involved[i]]
    
    def size(self):
        return self.aoR.nbytes + self.ao_involved.nbytes + self.segments.nbytes

    def todense(self, nao):
        aoR = np.zeros((nao, self.aoR.shape[1]))
        aoR[self.ao_involved] = self.aoR
        return aoR

class aoPairR_Holder:
    
    def __init__(self, aoPairR, ao_involved, grid_involved):
        assert aoPairR.shape[1] == len(ao_involved)
        assert aoPairR.shape[2] == len(grid_involved) 
        
        self.aoPairR = aoPairR
        self.ao_involved = np.array(ao_involved, dtype=np.int32)
        self.grid_involved = np.array(grid_involved, dtype=np.int32)
        self.nao_given_atm = aoPairR.shape[0]
        self.nao_involved  = len(ao_involved)
        self.nGrid_involved = len(grid_involved)

    def size(self):
        return self.aoPairR.nbytes + self.ao_involved.nbytes + self.grid_involved.nbytes
    
    def todense(self, nao, nGrid):
        aoPairR = np.zeros((self.nao_given_atm, self.nao_involved, nGrid))
        aoPairR[:,:, self.grid_involved] = self.aoPairR
        res = np.zeros((self.nao_given_atm, nao, nGrid))
        res[:, self.ao_involved,:] = aoPairR
        return res

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
    return sum([_aoR_holder.size() for _aoR_holder in aoR_holders if _aoR_holder is not None])

def _pack_aoR_holder(aoR_holders:list[aoR_Holder], nao):
    has_involved = [False] * nao
    
    nGrid = 0
    for _aoR_holder in aoR_holders:
        if _aoR_holder is None:
            continue
        for i in _aoR_holder.ao_involved:
            has_involved[i] = True  
        # nGrid += _aoR_holder.aoR.shape[1]
        nGrid += _aoR_holder.ngrid_tot
    
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
        if _aoR_holder is None:
            continue
        loc_packed = np.zeros((_aoR_holder.aoR.shape[0]), dtype=np.int32)
        # grid_end_id = grid_begin_id + _aoR_holder.aoR.shape[1]
        grid_end_id = grid_begin_id + _aoR_holder.ngrid_tot
        for loc, ao_id in enumerate(_aoR_holder.ao_involved):
            loc_packed[loc] = ao2loc[ao_id]
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
                  Ls=[3,3,3], 
                  with_translation_symmetry=False,
                  kmesh=None,
                  use_mpi=False): # by default split the cell into 4x4x4 supercell
    
    ##### this step is super fast #####
    
    ##### we simply perform it on root and broadcast it to all other processes #####
    
    if use_mpi:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, allgather_list, bcast_pickel
    
    if with_translation_symmetry and kmesh is None:
        raise ValueError("kmesh must be provided if with_translation_symmetry is True")
    
    log = lib.logger.Logger(cell.stdout, cell.verbose)
    
    if use_mpi == False or (use_mpi and rank == 0):
        #print("************* get_partition *************")
        log.debug4("************* get_partition *************")
    
    ##### construct the box info #####

    mesh = cell.mesh
    lattice_vector = cell.lattice_vectors()
    lattice_vector = np.array(lattice_vector)
    
    meshPrim = None
    if with_translation_symmetry:
        meshPrim = np.array(mesh) // np.array(kmesh)
    
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
        
    Ls_box = [lattice_vector[0] / mesh[0] * mesh_box[0], lattice_vector[1] / mesh[1] * mesh_box[1], lattice_vector[2] / mesh[2] * mesh_box[2]]
    
    print("Ls       = ", Ls)
    print("mesh     = ", mesh)
    print("mesh_box = ", mesh_box)
    print("Ls_box   = ", Ls_box)
        
    assert Ls_box[0][0] < 3.0
    assert Ls_box[1][1] < 3.0
    assert Ls_box[2][2] < 3.0 # the box cannot be too large
    
    ##### helper functions ##### 
    
    def get_box_id(x, y, z):
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
    
    atm_coords = []

    for i in range(cell.natm):
        box_id = get_box_id_from_coord(cell.atom_coord(i))
        atm_box_id.append(box_id)
        if box_id not in box_2_atm:
            box_2_atm[box_id] = [i]
        else:
            box_2_atm[box_id].append(i)
        atm_coords.append(cell.atom_coord(i))

    atm_coords = np.array(atm_coords)
    distance = np.zeros((cell.natm,), dtype=np.float64)
    
    fn_calculate_distance = getattr(libpbc, "distance_between_point_atms", None)
    assert fn_calculate_distance is not None

    fn_calculate_distance2 = getattr(libpbc, "distance_between_points_atms", None)
    assert fn_calculate_distance2 is not None

    ######## a rough partition of the cell based on distance only ######## 
    
    natm_tmp = cell.natm
    if with_translation_symmetry:
        natm_tmp = cell.natm // np.prod(kmesh)
    partition_rough = []
    for i in range(natm_tmp):
        partition_rough.append([])

    grid_id_global = np.arange(mesh[0] * mesh[1] * mesh[2], dtype=np.int32).reshape(mesh[0], mesh[1], mesh[2])

    for ix in range(nbox[0]):
        for iy in range(nbox[1]):
            for iz in range(nbox[2]):
                
                if use_mpi and rank != 0:
                    continue
                
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
            
                IsValidBox=True
                if with_translation_symmetry:
                    if mesh_x_begin >= meshPrim[0]:
                        IsValidBox=False
                    if mesh_y_begin >= meshPrim[1]:
                        IsValidBox=False
                    if mesh_z_begin >= meshPrim[2]:
                        IsValidBox=False
                if not IsValidBox:
                    continue
                
                if with_translation_symmetry:
                    mesh_x_end = min(mesh_x_end, meshPrim[0])
                    mesh_y_end = min(mesh_y_end, meshPrim[1])
                    mesh_z_end = min(mesh_z_end, meshPrim[2])

                grid_ID = grid_id_global[mesh_x_begin:mesh_x_end, mesh_y_begin:mesh_y_end, mesh_z_begin:mesh_z_end].flatten()
                
                grid_ID.sort()
                grid_ID = np.array(grid_ID, dtype=np.int32)
                
                # print("grid_ID = ", grid_ID)
                
                if box_id in box_2_atm:
                    partition_rough[box_2_atm[box_id][0]%natm_tmp].extend(grid_ID)
                else:
                    # random pickup one coord in the box #
                    
                    grid_ID_random_pick = grid_ID[np.random.randint(0, len(grid_ID))]
                    grid_coord = coords[grid_ID_random_pick]
                    grid_coord = np.array(grid_coord)
                    
                    fn_calculate_distance(
                        distance.ctypes.data_as(ctypes.c_void_p),
                        grid_coord.ctypes.data_as(ctypes.c_void_p),
                        atm_coords.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(cell.natm),
                        lattice_vector.ctypes.data_as(ctypes.c_void_p)
                    )
                    
                    atm_id = np.argmin(distance)
                    partition_rough[atm_id%natm_tmp].extend(grid_ID)
      
    if use_mpi:
        comm.Barrier()
            
    if use_mpi == False or (use_mpi == True and rank == 0):
        len_grid_involved = 0
        for atm_id, x in enumerate(partition_rough):
            print("atm %d involved %d grids" % (atm_id, len(x)))
            len_grid_involved += len(x)
        if with_translation_symmetry:
            assert len_grid_involved == np.prod(mesh) // np.prod(kmesh)
        else:
            assert len_grid_involved == mesh[0] * mesh[1] * mesh[2]
    
    ######## refine the partition based on the AtmConnectionInfo ########
    
    partition = []
    natm_tmp = cell.natm
    if with_translation_symmetry:
        natm_tmp = cell.natm // np.prod(kmesh)
        assert cell.natm % np.prod(kmesh) == 0
    for i in range(natm_tmp):
        partition.append([])
    
    ao_loc = cell.ao_loc_nr()
    # print("nao_intot = ", ao_loc[-1])
    
    from copy import deepcopy
    lattice_vector = deepcopy(cell.lattice_vectors())
    
    # print("lattice_vector = ", lattice_vector)
    
    if with_translation_symmetry:
        # print("lattice_vector = ", lattice_vector)
        lattice_vector = np.array(lattice_vector) / np.array(kmesh)
        # print("lattice_vector = ", lattice_vector)
    
    for atm_id in range(natm_tmp):
        
        atm_involved = []
        
        if use_mpi and rank != 0:
            continue
        
        ## pick up atms with distance < DISTANCE_CUTOFF ##
        
        for atm_id_other, distance in AtmConnectionInfoList[atm_id].atm_connected_info:
            # print("atm %d distance = %f" % (atm_id_other, distance))
            if distance < DISTANCE_CUTOFF:
                atm_involved.append(atm_id_other % natm_tmp)
            if len(atm_involved) >= 16: ## up to 16 atms 
                break 
        atm_involved.sort()
        atm_involved = list(set(atm_involved))
        atm_involved = np.array(atm_involved, dtype=np.int32)
        # print("atm %d involved atm = %s" % (atm_id, atm_involved))
        
        ## get the involved ao ##
        
        atm_coords_involved = []
        
        nao_involved = 0
        for atm_id_other in atm_involved:
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end = AtmConnectionInfoList[atm_id_other].bas_range[-1]+1
            nao_involved += ao_loc[shl_end] - ao_loc[shl_begin]
            atm_coords_involved.append(cell.atom_coord(atm_id_other))
        
        atm_coords_involved = np.array(atm_coords_involved)
        
        grid_ID = partition_rough[atm_id]
                    
        ## determine the partition by distance ##
        
        coords_now = coords[grid_ID].copy()
        distance = np.zeros((len(grid_ID), len(atm_involved)), dtype=np.float64)
        fn_calculate_distance2(
            distance.ctypes.data_as(ctypes.c_void_p),
            coords_now.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(grid_ID)),
            atm_coords_involved.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(atm_involved)),
            lattice_vector.ctypes.data_as(ctypes.c_void_p)
        )
        argmin_distance = np.argmin(distance, axis=1)
        for grid_id, _atm_id_ in zip(grid_ID, argmin_distance):
            partition[atm_involved[_atm_id_]%natm_tmp].append(grid_id)
    
    if use_mpi == False or (use_mpi == True and rank == 0):
        len_grid_involved = 0
        for atm_id, x in enumerate(partition):
            len_grid_involved += len(x)
        if with_translation_symmetry:
            assert len_grid_involved == np.prod(mesh) // np.prod(kmesh)
        else:
            assert len_grid_involved == mesh[0] * mesh[1] * mesh[2]
    
    del partition_rough
    
    if use_mpi:
        partition_sendbuf = [np.array(x, dtype=np.int32) for x in partition]
        partition = []
        for x in partition_sendbuf:
            partition.append(bcast(x))
        del partition_sendbuf
    
    if (use_mpi and rank == 0) or use_mpi == False:
        #print("************* end get_partition *************")
        log.debug4("************* end get_partition *************")
    
    return partition

def _range_partition(ngroup, rank, comm_size, use_mpi=False):
    if use_mpi == False:
        return 0, ngroup
    else:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import comm_size
        if ngroup % comm_size == 0:
            ngroup_local = ngroup // comm_size
            return rank * ngroup_local, (rank+1) * ngroup_local
        else:
            ngroup_local = ngroup // comm_size + 1
            
            ## solve equation a * ngroup_local + b * (ngroup_local - 1) = ngroup ## 
            ## a + b = comm_size ##
            
            b = (ngroup_local * comm_size - ngroup)    
            a = comm_size - b
            
            if rank < a:
                return rank * ngroup_local, (rank+1) * ngroup_local
            else:
                return a * ngroup_local + (rank - a) * (ngroup_local - 1), a * ngroup_local + (rank - a + 1) * (ngroup_local - 1)

def _range_partition_array(ngroup, comm_size, use_mpi=False):
    if use_mpi == False:
        return np.array([0, ngroup], dtype=np.int32)
    else:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import comm_size
        if ngroup % comm_size == 0:
            ngroup_local = ngroup // comm_size
            for i in range(comm_size):
                if i == 0:
                    res = np.array([0, ngroup_local], dtype=np.int32)
                else:
                    res = np.vstack((res, np.array([i * ngroup_local, (i+1) * ngroup_local], dtype=np.int32)))
        else:
            ngroup_local = ngroup // comm_size + 1
            
            ## solve equation a * ngroup_local + b * (ngroup_local - 1) = ngroup ## 
            ## a + b = comm_size ##
            
            b = (ngroup_local * comm_size - ngroup)    
            a = comm_size - b
            
            for i in range(comm_size):
                if i < a:
                    if i == 0:
                        res = np.array([0, ngroup_local], dtype=np.int32)
                    else:
                        res = np.vstack((res, np.array([i * ngroup_local, (i+1) * ngroup_local], dtype=np.int32)))
                else:
                    if i == a:
                        res = np.vstack((res, np.array([a * ngroup_local, a * ngroup_local + (ngroup_local - 1)], dtype=np.int32)))
                    else:
                        res = np.vstack((res, np.array([a * ngroup_local + (i - a) * (ngroup_local - 1), a * ngroup_local + (i - a + 1) * (ngroup_local - 1)], dtype=np.int32)))

        if comm_size == 1:
            res = res.reshape(1, 2)
        return res

def _get_grid_ordering(atmid_to_gridID, group, use_mpi=False):
        
    grid_ordering = []
    for i in range(len(group)):
        for atmid in group[i]:
            grid_ordering.extend(atmid_to_gridID[atmid])
        
    return np.array(grid_ordering, dtype=np.int32)

def _get_grid_partition(atmid_to_gridID, group, use_mpi=False):
    
    if use_mpi:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import comm_size
    
    ngrid = np.sum([len(x) for x in atmid_to_gridID])
    
    if use_mpi == False:
        return np.array([0, ngrid], dtype=np.int32)
    else:
        group_partition_array = _range_partition_array(len(group), comm_size, use_mpi)
        
        grid_partition = [0]
        for i in range(comm_size):
            group_begin = group_partition_array[i][0]
            group_end = group_partition_array[i][1]
            
            ngrid_local = 0
            for j in range(group_begin, group_end):
                for atmid in group[j]:
                    ngrid_local += len(atmid_to_gridID[atmid])
            
            grid_partition.append(grid_partition[-1] + ngrid_local)
        
        return np.array(grid_partition, dtype=np.int32)

def _get_atm_2_grid_segment(atmid_to_gridID, group):

    natm = len(atmid_to_gridID)
    assert sum([len(x) for x in group]) == natm or (natm % sum([len(x) for x in group])) == 0
    
    res = []
    for _ in range(natm):
        res.append([None, None])
        
    grid_loc_now = 0
    for j in range(len(group)):
        for atmid in group[j]:
            res[atmid][0] = grid_loc_now
            res[atmid][1] = grid_loc_now + len(atmid_to_gridID[atmid])
            grid_loc_now += len(atmid_to_gridID[atmid])
    
    return res
    
def _get_atmid_involved(natm, group, rank, use_mpi=False):
    if use_mpi == False:
        return np.arange(natm, dtype=np.int32)
    else:
        group_partition_array = _range_partition_array(len(group), comm_size, use_mpi)
                
        atmid_involved = []
        group_begin = group_partition_array[rank][0]
        group_end = group_partition_array[rank][1]
        for i in range(group_begin, group_end):
            atmid_involved.extend(group[i])
        
        return np.array(atmid_involved, dtype=np.int32)

def _sync_list(list_data, ngroup):

    # if use_mpi:
    from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm_size, bcast

    ### check data ### 
    
    if len(list_data) != ngroup:
        raise ValueError("the length of list_data is not equal to ngroup")
    
    group_begin, group_end = _range_partition(ngroup, rank, comm_size, True)
    
    for i in range(group_begin):
        assert list_data[i] is None
    for i in range(group_end, ngroup):
        assert list_data[i] is None
    for i in range(group_begin, group_end):
        assert list_data[i] is not None
    
    ### generate groupid_2_root ###
    
    groupid_2_root = [] 
    
    range_partition_array = _range_partition_array(ngroup, comm_size, True) 
    
    for j in range(comm_size):
        group_begin = range_partition_array[j][0]
        group_end = range_partition_array[j][1]
        for i in range(group_begin, group_end):
            groupid_2_root.append(j)
    
    ### sync ### 
    
    for i in range(ngroup):
        if rank == groupid_2_root[i]:
            sys.stdout.flush()
        list_data[i] = bcast(list_data[i], root=groupid_2_root[i])
        
    return list_data

def _sync_list_related_to_partition(list_data, group):

    natm = sum([len(x) for x in group])
    assert len(list_data) == natm
    
    group_begin, group_end = _range_partition(len(group), rank, comm_size, True)
    
    atm_involved = []
    for i in range(group_begin, group_end):
        atm_involved.extend(group[i])
    atm_involved = np.array(atm_involved, dtype=np.int32)
    atm_involved.sort()
    
    for i in range(natm):
        if i in atm_involved:
            assert list_data[i] is not None
        else:
            assert list_data[i] is None
    
    atmid_2_root = [0] * natm
    
    range_partition_array = _range_partition_array(len(group), comm_size, True)
    
    for j in range(comm_size):
        group_begin = range_partition_array[j][0]
        group_end = range_partition_array[j][1]
        for i in range(group_begin, group_end):
            for atmid in group[i]:
                atmid_2_root[atmid] = j

    #### sync ####
    
    for i in range(natm):
        list_data[i] = bcast_pickel(list_data[i], root=atmid_2_root[i])

    return list_data

def _sync_aoR(aoR_holders, natm):
    
    # return _sync_list_related_to_partition(aoR_holders, group)

    aoR = []
    bas_id = []
    grid_ID_begin = []
    for i in range(natm):
        if aoR_holders[i] is not None:
            aoR.append(aoR_holders[i].aoR)
            bas_id.append(aoR_holders[i].ao_involved)
            grid_ID_begin.append(np.asarray([aoR_holders[i].global_gridID_begin],dtype=np.int32))
        else:
            aoR.append(None)
            bas_id.append(None)
            grid_ID_begin.append(None)

    aoR = _sync_list(aoR, natm)
    bas_id = _sync_list(bas_id, natm)
    grid_ID_begin = _sync_list(grid_ID_begin, natm)

    aoR_holders = []
    
    for i in range(natm):
        aoR_holders.append(
            aoR_Holder(aoR[i], bas_id[i], grid_ID_begin[i][0], grid_ID_begin[i][0] + aoR[i].shape[1], grid_ID_begin[i][0], grid_ID_begin[i][0] + aoR[i].shape[1])
        )

    return aoR_holders

def _build_submol(cell:Cell, atm_invovled):
    
    import pyscf.pbc.gto as pbcgto
    
    subcell = pbcgto.Cell()
    subcell.a = cell.a
    
    atm = []
    for atm_id in atm_invovled:
        atm.append(cell.atom[atm_id])
    
    subcell.atom = atm
    subcell.basis = cell.basis
    subcell.pseudo = cell.pseudo
    subcell.verbose = 0
    subcell.ke_cutoff = cell.ke_cutoff
    subcell.max_memory = cell.max_memory
    subcell.precision = cell.precision
    subcell.use_particle_mesh_ewald = cell.use_particle_mesh_ewald
    subcell.mesh = cell.mesh
    subcell.unit = cell.unit
    subcell.build(mesh = cell.mesh)
    
    return subcell

def get_aoR(cell:Cell, coords, partition, 
            first_npartition = None,
            first_natm=None, group=None, 
            distance_matrix=None, AtmConnectionInfoList:list[AtmConnectionInfo]=None, 
            distributed = False, use_mpi=False, sync_res = False):
    
    if first_natm is None:
        first_natm = cell.natm
    if first_npartition is None:
        first_npartition = len(partition)
    
    ## aoR is stored distributedly ##
    
    log = lib.logger.Logger(cell.stdout, cell.verbose)
    
    if use_mpi:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, allgather_list, bcast_pickel
        if rank == 0:
            log.debug4("************* get_aoR *************")
    else:
        rank = 0
        comm_size = 1
        log.debug4("************* get_aoR *************")
    
    weight = np.sqrt(cell.vol / coords.shape[0])
    
    RcutMax = -1e10
    
    for _info_ in AtmConnectionInfoList:
        RcutMax = max(RcutMax, np.max(_info_.bas_cut))
    
    precision = AtmConnectionInfoList[0].precision
        
    aoR_holder = []
    
    if group == None:
        group = []
        for i in range(cell.natm):
            group.append([i])
    
    for _ in range(first_npartition):
        aoR_holder.append(None)
    
    grid_partition = _get_grid_partition(partition, group, use_mpi)
    
    atm_2_grid_segment = _get_atm_2_grid_segment(partition, group)
    
    local_gridID_begin = 0
    global_gridID_begin = grid_partition[rank]
    ao_loc = cell.ao_loc_nr()
    
    atm_begin, atm_end = _range_partition(first_npartition, rank, comm_size, use_mpi)

    for atm_id in range(atm_begin, atm_end):
        
        grid_ID = partition[atm_id]
        
        if len(grid_ID) == 0:
            aoR_holder[atm_id] = None
            continue
        
        ##### find the involved atms within RcutMax #####
        
        if first_natm!=cell.natm:
            atm_involved = np.arange(first_natm) # with kmesh ! 
        else:
            if first_npartition == len(partition):
                atm_involved = []
                for atm_id_other, distance in AtmConnectionInfoList[atm_id].atm_connected_info:
                    if distance < RcutMax and atm_id_other < first_natm:
                        atm_involved.append(atm_id_other)
                atm_involved.sort()
            else:
                atm_involved = np.arange(cell.natm) # with kmesh ! 
                
        ##### get the involved ao #####
        
        nao_involved = 0
        for atm_id_other in atm_involved:
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end = AtmConnectionInfoList[atm_id_other].bas_range[-1]+1
            nao_involved += ao_loc[shl_end] - ao_loc[shl_begin]
        
        bas_id = []

        ao_loc_now = 0
        
        shell_slice = []
        shl_end_test = 0
        for atm_id_other in atm_involved:
            shl_begin = AtmConnectionInfoList[atm_id_other].bas_range[0]
            shl_end = AtmConnectionInfoList[atm_id_other].bas_range[-1]+1
            bas_id.extend(np.arange(ao_loc[shl_begin], ao_loc[shl_end]))
        
        bas_id = np.array(bas_id)
                
        subcell = _build_submol(cell, atm_involved)
        aoR     = ISDF_eval_gto(subcell, coords=coords[grid_ID]) * weight
        
        assert aoR.shape[0] == len(bas_id)
        
        ##### screening the aoR, TODO: in C ##### 
        
        max_row = np.max(np.abs(aoR), axis=1)
        where = np.where(max_row > precision)[0]
        if len(where) < aoR.shape[0] * 0.9:
            aoR = aoR[where]
            bas_id = np.array(bas_id)[where]
        
        global_gridID_begin = atm_2_grid_segment[atm_id][0]
        aoR_holder[atm_id]  = aoR_Holder(aoR, bas_id, local_gridID_begin, local_gridID_begin+len(grid_ID), global_gridID_begin, global_gridID_begin+len(grid_ID))
        
        assert global_gridID_begin == atm_2_grid_segment[atm_id][0]
        assert global_gridID_begin + len(grid_ID) == atm_2_grid_segment[atm_id][1]
        
        local_gridID_begin += len(grid_ID)
        global_gridID_begin += len(grid_ID)
                        
    del aoR
    
    if use_mpi and sync_res:
        aoR_holder = _sync_aoR(aoR_holder, cell.natm)
        
    if use_mpi:
        if rank == 0:
            log.debug4("************* end get_aoR *************")
    else:
        log.debug4("************* end get_aoR *************")
    
    return aoR_holder

def get_aoR_analytic(cell:Cell, coords, partition, 
                     first_npartition = None,
                     first_natm=None, group=None, 
                     distance_matrix=None, AtmConnectionInfoList:list[AtmConnectionInfo]=None, 
                     distributed = False, use_mpi=False, sync_res = False):
    
    assert use_mpi == False
    assert first_natm is None or first_natm == cell.natm
    
    if group is None:
        group = []
        for i in range(cell.natm):
            group.append([i])
    
    precision = AtmConnectionInfoList[0].precision
    mesh = cell.mesh
    ngrids = np.prod(mesh)
    weight = cell.vol/ngrids
    weight2 = np.sqrt(cell.vol / ngrids)
    
    blksize = 2e9//16
    nao_max_bunch = int(blksize // ngrids)

    Gv = cell.get_Gv() 

    ######## pack info ########
    
    aoR_unpacked = []
    ao_invovled_unpacked = []
    atm_ordering = []
    for group_idx in group:
        group_idx.sort()
        atm_ordering.extend(group_idx)
    grid_begin_unpacked = []
    grid_end_unpacked = []
    grid_ID_now = 0
    for atm_id in atm_ordering:
        grid_ID = partition[atm_id]
        grid_begin_unpacked.append(grid_ID_now)
        grid_end_unpacked.append(grid_ID_now + len(grid_ID))
        grid_ID_now += len(grid_ID)
        aoR_unpacked.append([])
        ao_invovled_unpacked.append([])
    
    ao_loc = cell.ao_loc_nr()
    
    task_sl_loc = [0] 
    ao_loc_now = 0
    for i in range(cell.nbas):
        ao_loc_end = ao_loc[i+1]
        if ao_loc_end - ao_loc_now > nao_max_bunch:
            task_sl_loc.append(i)
            ao_loc_now = ao_loc[i]
    task_sl_loc.append(cell.nbas)
    print("task_sl_loc = ", task_sl_loc)
    nTask = len(task_sl_loc) - 1
    print("nTask       = ", nTask)
    
    for task_id in range(nTask):
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        shloc    = (task_sl_loc[task_id], task_sl_loc[task_id+1])
        aoG      = ft_ao.ft_ao(cell, Gv, shls_slice=shloc).T
        
        ### implementation 1 ###
        # aoR_test = numpy.fft.ifftn(aoG.reshape(-1, *mesh), axes=(1,2,3)).real / (weight)
        # aoR = aoR_test.reshape(-1, ngrids) * weight2
        
        ### implementation 2 ###
        aoR_test = None
        aoG = aoG.conj() * np.sqrt(1/cell.vol)
        aoG = aoG.reshape(-1, *mesh)
        aoR = numpy.fft.fftn(aoG, axes=(1,2,3)).real * np.sqrt(1/float(ngrids))
        aoR = aoR.reshape(-1, ngrids)
        
        bas_id = np.arange(ao_loc[shloc[0]], ao_loc[shloc[1]])
        
        for atm_id, atm_partition in enumerate(partition):
            aoR_tmp = aoR[:, atm_partition].copy()
            ### prune the aoR ### 
            where = np.where(np.max(np.abs(aoR_tmp), axis=1) > precision)[0]
            aoR_tmp = aoR_tmp[where].copy()
            bas_id_tmp = bas_id[where].copy()
            aoR_unpacked[atm_id].append(aoR_tmp)
            ao_invovled_unpacked[atm_id].append(bas_id_tmp)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if rank == 0:
            _benchmark_time(t1, t2, "get_aoR_analytic: task %d" % task_id)

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    aoR_holder = []
    
    for atm_id in range(len(aoR_unpacked)):
        aoR_holder_tmp = np.concatenate(aoR_unpacked[atm_id], axis=0)
        bas_id =         np.concatenate(ao_invovled_unpacked[atm_id], axis=0) 
        aoR_holder.append(aoR_Holder(aoR_holder_tmp, bas_id, grid_begin_unpacked[atm_id], grid_end_unpacked[atm_id], grid_begin_unpacked[atm_id], grid_end_unpacked[atm_id]))
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    del aoR_unpacked
    del ao_invovled_unpacked
    del aoR_tmp
    del aoR_holder_tmp
    del bas_id
    del aoR_test
    del aoR
    del aoG
    
    
    if rank == 0:
        _benchmark_time(t1, t2, "get_aoR_analytic: merge")

    return aoR_holder

def get_aoR_exact(cell:Cell, 
                  # coords, 
                  mesh_sparse,
                  mesh_dense,
                  partition, 
                  first_npartition = None,
                  first_natm=None, 
                  group=None, 
                  distance_matrix=None, AtmConnectionInfoList:list[AtmConnectionInfo]=None, 
                  distributed = False, use_mpi=False, sync_res = False):
    
    assert use_mpi == False
    assert first_natm is None or first_natm == cell.natm
    
    if group is None:
        group = []
        for i in range(cell.natm):
            group.append([i])
    
    precision = AtmConnectionInfoList[0].precision
    cell_sparse = cell.copy()
    cell_sparse.build(mesh = mesh_sparse)
    cell_dense = cell.copy()
    cell_dense.build(mesh = mesh_dense)
    
    #### some convention
    
    mesh = cell_dense.mesh
    ngrids = np.prod(mesh)
    weight = cell_dense.vol/ngrids
    weight2 = np.sqrt(cell_dense.vol / ngrids)

    blksize = 2e9//16
    nao_max_bunch = int(blksize // ngrids)

    Gv = cell_dense.get_Gv() 
    
    print("Gv.size = ", Gv.size)
    print("np.prod(mesh) = ", np.prod(mesh))
    assert Gv.shape[0] == np.prod(mesh)
    
    #### sparse grid ####
    
    mesh_sparse = cell_sparse.mesh
    ngrids_sparse = np.prod(mesh_sparse)
    assert mesh_dense[0] % mesh_sparse[0] == 0
    assert mesh_dense[1] % mesh_sparse[1] == 0
    assert mesh_dense[2] % mesh_sparse[2] == 0
    
    step = (mesh_dense[0] // mesh_sparse[0], mesh_dense[1] // mesh_sparse[1], mesh_dense[2] // mesh_sparse[2])
    
    weight_dense_to_sparse = np.sqrt(float(ngrids)/float(ngrids_sparse))
    
    precision = precision / weight_dense_to_sparse
    
    ######## pack info ########
    
    aoR_unpacked = []
    ao_invovled_unpacked = []
    atm_ordering = []
    for group_idx in group:
        group_idx.sort()
        atm_ordering.extend(group_idx)
    grid_begin_unpacked = []
    grid_end_unpacked = []
    grid_ID_now = 0
    for atm_id in atm_ordering:
        grid_ID = partition[atm_id]
        grid_begin_unpacked.append(grid_ID_now)
        grid_end_unpacked.append(grid_ID_now + len(grid_ID))
        grid_ID_now += len(grid_ID)
        aoR_unpacked.append([])
        ao_invovled_unpacked.append([])
    
    ao_loc = cell.ao_loc_nr()
    
    task_sl_loc = [0] 
    ao_loc_now = 0
    for i in range(cell.nbas):
        ao_loc_end = ao_loc[i+1]
        if ao_loc_end - ao_loc_now > nao_max_bunch:
            task_sl_loc.append(i)
            ao_loc_now = ao_loc[i]
    task_sl_loc.append(cell.nbas)
    print("task_sl_loc = ", task_sl_loc)
    nTask = len(task_sl_loc) - 1
    print("nTask       = ", nTask)
    
    for task_id in range(nTask):
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        shloc    = (task_sl_loc[task_id], task_sl_loc[task_id+1])
        aoG      = ft_ao.ft_ao(cell, Gv, shls_slice=shloc).T
        aoR_test = numpy.fft.ifftn(aoG.reshape(-1, *mesh), axes=(1,2,3)).real / (weight)
        # aoR = aoR_test.reshape(-1, ngrids) * weight2
        aoR = aoR_test * weight2
        
        bas_id = np.arange(ao_loc[shloc[0]], ao_loc[shloc[1]])
        
        aoR = aoR[:, ::step[0], ::step[1], ::step[2]] * weight_dense_to_sparse
        aoR = aoR.reshape(-1, ngrids_sparse)
        
        assert aoR.shape[1] == ngrids_sparse 
        
        print("aoR.shape = ", aoR.shape)
        
        for atm_id, atm_partition in enumerate(partition):
            aoR_tmp = aoR[:, atm_partition].copy()
            ### prune the aoR ### 
            where = np.where(np.max(np.abs(aoR_tmp), axis=1) > precision)[0]
            aoR_tmp = aoR_tmp[where].copy()
            bas_id_tmp = bas_id[where].copy()
            aoR_unpacked[atm_id].append(aoR_tmp)
            ao_invovled_unpacked[atm_id].append(bas_id_tmp)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if rank == 0:
            _benchmark_time(t1, t2, "get_aoR_analytic: task %d" % task_id)

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    aoR_holder = []
    
    for atm_id in range(len(aoR_unpacked)):
        aoR_holder_tmp = np.concatenate(aoR_unpacked[atm_id], axis=0)
        bas_id =         np.concatenate(ao_invovled_unpacked[atm_id], axis=0) 
        aoR_holder.append(aoR_Holder(aoR_holder_tmp, bas_id, grid_begin_unpacked[atm_id], grid_end_unpacked[atm_id], grid_begin_unpacked[atm_id], grid_end_unpacked[atm_id]))
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    del aoR_unpacked
    del ao_invovled_unpacked
    del aoR_tmp
    del aoR_holder_tmp
    del bas_id
    del aoR_test
    del aoR
    del aoG
    
    
    if rank == 0:
        _benchmark_time(t1, t2, "get_aoR_analytic: merge")

    return aoR_holder

def get_aoPairR_analytic(cell_c_input:Cell, 
                         ke_cutoff_in_rsjk,
                         mesh, 
                         distance_matrix=None, AtmConnectionInfoList:list[AtmConnectionInfo]=None, 
                         use_mpi=False,
                         compressed=False
                         ):
    
    if use_mpi:
        raise NotImplementedError("not implemented yet")
    
    cell_c = pyscf.pbc.df.ft_ao._RangeSeparatedCell.from_cell(cell_c_input, rsjk.ke_cutoff, in_rsjk=True)
    
    precision = AtmConnectionInfoList[0].precision
    rcut = _estimate_rcut(cell_c, np.prod(mesh), precision)
    supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(cell_c, [1,1,1], rcut.max())
    supmol_ft = supmol_ft.strip_basis(rcut)
    
    ngrids = np.prod(mesh)
    
    Gv, Gvbase, kws = cell_c.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    
    ## TODO: consider the weight ## 
    
    ### loop over all atm ###

if __name__ == '__main__':

    from pyscf.lib.parameters import BOHR

    TARGET_PRECISION = 1e-9
    
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
        'Cu1':'unc-ecpccpvdz', 'Cu2':'unc-ecpccpvdz', 'O1': 'unc-ecpccpvdz', 'Ca':'unc-ecpccpvdz'
    }
    # basis  = 'gth-cc-pvdz'
    pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}

    # basis = 'unc-ccpvdz'
    # pseudo = None
    
    ke_cutoff = 128
    
    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo, verbose=10)
    # prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=None, verbose=4)
    prim_mesh = prim_cell.mesh
    
    supercell = [2, 2, 1]
    # supercell = [4,4,2]
    
    mesh = [supercell[0] * prim_mesh[0], supercell[1] * prim_mesh[1], supercell[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell = build_supercell(atm, prim_a, Ls = supercell, ke_cutoff=ke_cutoff, mesh=mesh, basis=basis, pseudo=pseudo, verbose=10)
    # cell = build_supercell(atm, prim_a, Ls = supercell, ke_cutoff=ke_cutoff, mesh=mesh, basis=basis, pseudo=None, verbose=4)
    
    print(cell.atom)
    print(cell.basis)
    # exit(1)
    
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
    
    print("rcut = ", rcut)
    print("precision = ", precision)
    print("max_rcut = ", np.max(rcut))
    # print("nbas = ", cell.nbas)
    # print("number of rcut < 5 = ", np.sum(rcut < 5))
    # print("number of rcut < 15 = ", np.sum(rcut < 15))
    # print("number of rcut > 25 = ", np.sum(rcut > 25))
    # print("number of atm pair < 5 = ", np.sum(distance_matrix < 5))
    # print("number of atm pair < 15 = ", np.sum(distance_matrix < 15))
    # print("number of atm pair > rcut_max = ", np.sum(distance_matrix > rcut_max))
    
    exit(1)
    
    atm_2_bas = _atm_to_bas(cell)
    # print("atm_2_bas = ", atm_2_bas)
    # exit(1)
    AtmConnectionInfoList = []
    
    
    for i in range(cell.natm):
        tmp = AtmConnectionInfo(cell, i, distance_matrix, precision, rcut, rcut_max, atm_2_bas)
        # print(tmp)
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
        # print(tmp)
        AtmConnectionInfoList.append(tmp)
    
    print("comm_size = ", comm_size)
    # exit(1)
    
    if comm_size > 1:
        partition = get_partition(cell, coords, AtmConnectionInfoList, Ls=[supercell[0]*3, supercell[1]*3, supercell[2]*3], use_mpi=True)
        for i in range(comm_size):
            if i == rank:
                print("rank = %d" % rank)
                for _id_, _partition_ in enumerate(partition):
                    print("atm %d involved %d grids" % (_id_, len(_partition_)))
            comm.Barrier()
    else:
        partition = get_partition(cell, coords, AtmConnectionInfoList, Ls=[supercell[0]*3, supercell[1]*3, supercell[2]*3], use_mpi=False)    
        
    # exit(1)

    aoR_list = get_aoR(cell, coords, partition, 
                       distance_matrix=distance_matrix, 
                       AtmConnectionInfoList=AtmConnectionInfoList)
    
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