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

############ sys module ############

import copy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.gto.mole import *
libpbc = lib.load_library('libpbc')

############ isdf utils ############

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto 
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF_LinearScaling
import pyscf.pbc.df.isdf.isdf_tools_local as ISDF_Local_Utils
from pyscf.pbc.df.isdf.isdf_linear_scaling_k_jk import get_jk_dm_translation_symmetry
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

############ subroutines --- deal with translation symmetry ############

### WARNING: the unit cell must be put in the first cell !! ###

def _expand_partition_prim(partition_prim, kmesh, mesh):

    meshPrim = np.array(mesh) // np.array(kmesh) 
    
    partition = []
    
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                shift = i * meshPrim[0] * mesh[1] * mesh[2] + j * meshPrim[1] * mesh[2] + k * meshPrim[2]
                for data in partition_prim:
                    partition.append(data + shift)
    
    return partition

def _expand_primlist_2_superlist(primlist, kmesh, mesh):
    
    meshPrim = np.array(mesh) // np.array(kmesh)
    
    superlist = []
    
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                shift = i * meshPrim[0] * mesh[1] * mesh[2] + j * meshPrim[1] * mesh[2] + k * meshPrim[2]
                superlist.extend(primlist + shift)
    
    return np.array(superlist, dtype=np.int32)

def _get_grid_ordering_k(input, kmesh, mesh):
    
    if isinstance(input, list):
        prim_ordering = []
        for data in input:
            prim_ordering.extend(data)
        return _expand_primlist_2_superlist(prim_ordering, kmesh, mesh)
    else:
        raise NotImplementedError

def select_IP_local_ls_k_drive(mydf, c, m, IP_possible_atm, group, 
                               build_aoR_FFT=True,
                               use_mpi=False):
    
    assert use_mpi == False
    
    IP_group  = []
    aoRg_possible = mydf.aoRg_possible
    
    assert len(IP_possible_atm) == mydf.first_natm
    
    #### do the work ####
    
    first_natm = mydf.first_natm
    
    for i in range(len(group)):
        IP_group.append(None)

    if len(group) < first_natm:
        if use_mpi == False:
            for i in range(len(group)):
                IP_group[i] = ISDF_LinearScaling.select_IP_group_ls(
                    mydf, aoRg_possible, c, m,
                    group = group[i],
                    atm_2_IP_possible=IP_possible_atm
                )
        else:
            group_begin, group_end = ISDF_Local_Utils._range_partition(len(group), rank, comm_size, use_mpi)
            for i in range(group_begin, group_end):
                IP_group[i] = ISDF_LinearScaling.select_IP_local_ls(
                    mydf, aoRg_possible, c, m,
                    group = group[i],
                    atm_2_IP_possible=IP_possible_atm
                )
            IP_group = ISDF_Local_Utils._sync_list(IP_group, len(group))
    else:
        IP_group = IP_possible_atm

    mydf.IP_group = IP_group
    
    mydf.IP_flat_prim = []
    mydf.IP_segment_prim = []
    nIP_now = 0
    for x in IP_group:
        mydf.IP_flat_prim.extend(x)
        mydf.IP_segment_prim.append(nIP_now)
        nIP_now += len(x)
    mydf.IP_flat = _expand_primlist_2_superlist(mydf.IP_flat_prim, mydf.kmesh, mydf.mesh)
    mydf.naux = mydf.IP_flat.shape[0]
    # mydf.IP_segment = _expand_primlist_2_superlist(mydf.IP_segment_prim[:-1], mydf.kmesh, mydf.mesh)
    # mydf.IP_segment = np.append(mydf.IP_segment, mydf.naux)
    
    mydf.nIP_Prim = len(mydf.IP_flat_prim)
    mydf.nGridPrim = len(mydf.grid_ID_ordered_prim)
    
    gridID_2_atmID = mydf.gridID_2_atmID
    
    partition_IP = []
    for i in range(mydf.cell.natm):
        partition_IP.append([])
    
    for _ip_id_ in mydf.IP_flat:
        atm_id = gridID_2_atmID[_ip_id_]
        partition_IP[atm_id].append(_ip_id_)
    
    for i in range(mydf.cell.natm):
        partition_IP[i] = np.array(partition_IP[i], dtype=np.int32)
    
    mydf.IP_segment = [0]
    for atm_id in mydf.atm_ordering:
        mydf.IP_segment.append(mydf.IP_segment[-1] + len(partition_IP[atm_id]))
    mydf.IP_segment = np.array(mydf.IP_segment, dtype=np.int32)
    
    ### build aoR_IP ###
    
    #### recalculate it anyway ! #### 
    
    coords = mydf.coords
    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    
    del mydf.aoRg_possible
    mydf.aoRg_possible = None
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    
    mydf.aoRg = ISDF_Local_Utils.get_aoR(
        mydf.cell, coords, partition_IP,
        first_natm,
        mydf.cell.natm,
        mydf.group_global,
        mydf.distance_matrix,
        mydf.AtmConnectionInfo,
        False,
        mydf.use_mpi,
        True)
    
    assert len(mydf.aoRg) == first_natm
    
    mydf.aoRg1 = ISDF_Local_Utils.get_aoR(
        mydf.cell, coords, partition_IP,
        mydf.cell.natm,
        first_natm,
        mydf.group_global,
        mydf.distance_matrix,
        mydf.AtmConnectionInfo,
        False,
        mydf.use_mpi,
        True)
    
    # assert len(mydf.aoRg1) == mydf.cell.natm
    
    # mydf.aoRg1 = ISDF_Local_Utils.get_aoR(
    #     mydf.cell, coords, partition_IP,
    #     first_natm,
    #     mydf.group,
    #     mydf.distance_matrix,
    #     mydf.AtmConnectionInfo,
    #     False,
    #     mydf.use_mpi,
    #     True)
    
    aoRg_activated = []
    for _id_, aoR_holder in enumerate(mydf.aoRg):
        if aoR_holder.ao_involved.size == 0:
            aoRg_activated.append(False)
        else:
            aoRg_activated.append(True)
    aoRg_activated = np.array(aoRg_activated, dtype=bool)
    mydf.aoRg_activated = aoRg_activated
        
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    #################### build aoRg_FFT ####################

    kmesh = mydf.kmesh
    ncell_complex = kmesh[0] * kmesh[1] * (kmesh[2]//2+1)
    nao_prim = mydf.nao // np.prod(kmesh)
    nbas_prim = mydf.cell.nbas // np.prod(mydf.kmesh)
    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    nIP_Prim = mydf.nIP_Prim

    aoRg_Tmp = ISDF_eval_gto(mydf.cell, coords=coords[mydf.IP_flat], shls_slice=(0, nbas_prim)) * weight

    ### todo make it a list ! ### 
    
    ################# construct aoRg_FFT #################
    
    if build_aoR_FFT:
        
        mydf.aoRg_FFT  = np.zeros((nao_prim, ncell_complex*mydf.nIP_Prim), dtype=np.complex128)
        mydf.aoRg_FFT_real = np.ndarray((nao_prim, np.prod(kmesh)*mydf.nIP_Prim), dtype=np.double, buffer=mydf.aoRg_FFT, offset=0)
        mydf.aoRg_FFT_real.ravel()[:] = aoRg_Tmp.ravel()
    
        del aoRg_Tmp
        
        nthread        = lib.num_threads()
        buffer         = np.zeros((nao_prim, ncell_complex*mydf.nIP_Prim), dtype=np.complex128)
        
        fn = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
        assert fn is not None
    
        '''
        fn = _FFT_Matrix_Col_InPlace transform 

        (A0 | A1 | A2) --> (A0+A1+A2 | A0+wA1 + w^2 A2 | A0 + w^2 A1+ w A2)

        '''
        
        # print("aoRg_FFT.shape = ", mydf.aoRg_FFT.shape)
    
        kmesh = np.array(kmesh, dtype=np.int32)
    
        fn(
            mydf.aoRg_FFT_real.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_prim),
            ctypes.c_int(nIP_Prim),
            kmesh.ctypes.data_as(ctypes.c_void_p),
            buffer.ctypes.data_as(ctypes.c_void_p)
        ) # no normalization factor ! 

        aoRg_packed = []
        for i in range(ncell_complex):
            aoRg_packed.append(mydf.aoRg_FFT[:, i*nIP_Prim:(i+1)*nIP_Prim].copy())
        del mydf.aoRg_FFT
        mydf.aoRg_FFT = aoRg_packed
    else:
        mydf.aoRg_FFT = None
        # build aoRg #

    ################# End aoRg_FFT #################

    #################### build aoR_FFT ####################

    if mydf.with_robust_fitting and build_aoR_FFT:
        
        ngrids            = coords.shape[0]
        ngrids_prim       = ngrids // np.prod(kmesh)
        aoR_tmp           = ISDF_eval_gto(mydf.cell, coords=coords[mydf.grid_ID_ordered], shls_slice=(0, nbas_prim)) * weight
        mydf.aoR_FFT      = np.zeros((nao_prim, ncell_complex*ngrids_prim), dtype=np.complex128)
        mydf.aoR_FFT_real = np.ndarray((nao_prim, np.prod(kmesh)*ngrids_prim), dtype=np.double, buffer=mydf.aoR_FFT, offset=0)
        mydf.aoR_FFT_real.ravel()[:] = aoR_tmp.ravel()
        
        del aoR_tmp
        
        buffer         = np.zeros((nao_prim, ncell_complex*ngrids_prim), dtype=np.complex128)
        
        fn(
            mydf.aoR_FFT_real.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_prim),
            ctypes.c_int(ngrids_prim),
            kmesh.ctypes.data_as(ctypes.c_void_p),
            buffer.ctypes.data_as(ctypes.c_void_p)
        )

        aoR_packed = []
        for i in range(ncell_complex):
            aoR_packed.append(mydf.aoR_FFT[:, i*ngrids_prim:(i+1)*ngrids_prim].copy())
        del mydf.aoR_FFT
        mydf.aoR_FFT = aoR_packed
        # mydf.aoR     = None
        del buffer         
    else:
        mydf.aoR_FFT = None
        # build aoR #

def build_auxiliary_Coulomb_local_bas_k(mydf, debug=True, use_mpi=False):
    
    if use_mpi:
        raise NotImplementedError
    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = mydf.mesh
    
    naux = mydf.naux
    
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2 
    
    grid_ordering = mydf.grid_ID_ordered
    
    assert mydf.omega is None or mydf.omega == 0.0
    coulG = tools.get_coulG(cell, mesh=mesh)
    mydf.coulG = coulG.copy()
    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
    nThread = lib.num_threads()
    bufsize_per_thread = int((coulG_real.shape[0] * 2 + mesh[0] * mesh[1] * mesh[2]) * 1.1)
    buf = np.empty((nThread, bufsize_per_thread), dtype=np.double)
    
    def construct_V_CCode(aux_basis:list[np.ndarray], 
                          # buf:np.ndarray, 
                          V=None, shift_row=None):
        
        nThread = buf.shape[0]
        bufsize_per_thread = buf.shape[1]
        
        nAux = 0
        for x in aux_basis:
            nAux += x.shape[0]
        
        ngrids             = mesh[0] * mesh[1] * mesh[2]
        mesh_int32         = np.array(mesh, dtype=np.int32)

        if V is None:
            assert shift_row is None
            V = np.zeros((nAux, ngrids), dtype=np.double)
                    
        fn = getattr(libpbc, "_construct_V_local_bas", None)
        assert(fn is not None)

        if shift_row is None:
            shift_row = 0
        # ngrid_now = 0
        
        for i in range(len(aux_basis)):
            
            aux_basis_now = aux_basis[i]
            grid_ID = mydf.partition_group_to_gridID[i]
            # ngrid_now += grid_ID.size
            # print("i           = ", i)
            # print("shift_row   = ", shift_row) 
            # print("aux_bas_now = ", aux_basis_now.shape)
            # print("ngrid_now   = ", grid_ID.size)
            # print("buf = ", buf.shape)
            # print("ngrid_ordering = ", grid_ordering.size)
            # sys.stdout.flush()
            assert aux_basis_now.shape[1] == grid_ID.size 
        
            fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(aux_basis_now.shape[0]),
                ctypes.c_int(aux_basis_now.shape[1]),
                grid_ID.ctypes.data_as(ctypes.c_void_p),
                aux_basis_now.ctypes.data_as(ctypes.c_void_p),
                coulG_real.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(shift_row),
                V.ctypes.data_as(ctypes.c_void_p),
                grid_ordering.ctypes.data_as(ctypes.c_void_p),
                buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bufsize_per_thread))
        
            shift_row += aux_basis_now.shape[0]

        return V

    
    V = construct_V_CCode(mydf.aux_basis, V=None, shift_row=None)
    
    if mydf.with_robust_fitting:
        mydf.V_R = V
    
    ########### construct W ###########
    
    naux_bra = 0
    for x in mydf.aux_basis:
        naux_bra += x.shape[0]
    
    naux = mydf.naux
    
    assert naux % naux_bra == 0
    assert naux // naux_bra == np.prod(mydf.kmesh)
    
    mydf.W = np.zeros((naux_bra, naux), dtype=np.double)
    
    ngroup = len(mydf.aux_basis)    
    aux_bra_shift = 0
    kmesh = mydf.kmesh
        
    for i in range(ngroup):
            
        aux_ket_shift = 0
        grid_shift = 0
        naux_bra = mydf.aux_basis[i].shape[0]
        
        for ix in range(kmesh[0]):
            for iy in range(kmesh[1]):
                for iz in range(kmesh[2]):
                   for j in range(ngroup):
                        aux_basis_ket = mydf.aux_basis[j]
                        ngrid_now = aux_basis_ket.shape[1]
                        naux_ket = aux_basis_ket.shape[0]
                        mydf.W[aux_bra_shift:aux_bra_shift+naux_bra, aux_ket_shift:aux_ket_shift+naux_ket] = lib.ddot(
                           V[aux_bra_shift:aux_bra_shift+naux_bra, grid_shift:grid_shift+ngrid_now],
                           aux_basis_ket.T
                        )
                        aux_ket_shift += naux_ket
                        grid_shift += ngrid_now                 
                     
        aux_bra_shift += naux_bra
                        
        assert grid_shift == np.prod(mesh)
            
    del buf
    buf = None
    
    assert V.shape[0] == mydf.naux // np.prod(mydf.kmesh)
    assert V.shape[1] == np.prod(mesh)
    assert mydf.W.shape[0] == mydf.naux // np.prod(mydf.kmesh)
    assert mydf.W.shape[1] == mydf.naux
    
    if mydf.with_robust_fitting == False:
        del V
    
##### get_jk #####
    
class PBC_ISDF_Info_Quad_K(ISDF_LinearScaling.PBC_ISDF_Info_Quad):
    
    # Quad stands for quadratic scaling
    
    def __init__(self, 
                 mol:Cell,  # means the primitive cell 
                 with_robust_fitting=True,
                 kmesh              =None,
                 verbose            =None,
                 rela_cutoff_QRCP   =None,
                 aoR_cutoff         =1e-8,
                 direct             =False,
                 limited_memory     =False,
                 build_K_bunchsize  =None,
                 ):
        
        ### extract the info from the primitive cell ###
        
        atm = []
        
        assert mol.a[0][1] == 0.0
        assert mol.a[0][2] == 0.0
        assert mol.a[1][0] == 0.0
        assert mol.a[1][2] == 0.0
        assert mol.a[2][0] == 0.0
        assert mol.a[2][1] == 0.0
        
        from pyscf.lib.parameters import BOHR
        
        for i in range(mol.natm):
            coords = mol.atom_coord(i)
            coords = np.array(coords) * BOHR
            atm.append([mol.atom_symbol(i), tuple(coords)])
        
        prim_mesh = mol.mesh
        mesh = np.array(prim_mesh) * np.array(kmesh)
        
        nelectron = np.sum(mol.nelectron)
        
        from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell
        supercell = build_supercell(
            atm, 
            mol.a,
            spin=nelectron*np.prod(kmesh) % 2,
            mesh=mesh,
            Ls=kmesh,
            basis=mol.basis,
            pseudo=mol.pseudo,
            ke_cutoff=mol.ke_cutoff,
            max_memory=mol.max_memory,
            verbose=mol.verbose
        )
                
        self.prim_cell = mol
        
        # print("supercell.mesh = ", supercell.mesh)
        
        super().__init__(supercell, with_robust_fitting, None, verbose, rela_cutoff_QRCP, aoR_cutoff, direct, use_occ_RI_K=False, 
                         limited_memory=limited_memory, build_K_bunchsize=build_K_bunchsize)
        
        self.kmesh = kmesh
        
        self.kpts = self.prim_cell.make_kpts(kmesh)
        
        assert self.mesh[0] % kmesh[0] == 0
        assert self.mesh[1] % kmesh[1] == 0
        assert self.mesh[2] % kmesh[2] == 0
        
        # print("self.mesh = ", self.mesh)
        # exit(1)
        
        #### information relating primitive cell and supercell
        
        self.meshPrim = np.array(self.mesh) // np.array(self.kmesh)
        self.natm     = self.cell.natm
        self.natmPrim = self.cell.natm // np.prod(self.kmesh)
        
        self.with_translation_symmetry = True
        
        from pyscf.pbc.df.isdf.isdf_tools_cell import build_primitive_cell
        self.primCell = build_primitive_cell(self.cell, self.kmesh)
        self.nao_prim = self.nao // np.prod(self.kmesh)
        assert self.nao_prim == self.primCell.nao_nr()
    
        ##### rename everthing with pre_fix  _supercell ####
    
    def build_partition_aoR(self, Ls=None):
        '''
        
        build partition of grid points and AO values on grids 
        
        partition of grids is the assignment of each grids to the atom
        
        partition is hence a list of list of grids
        
        '''
        
        if self.aoR is not None and self.partition is not None:
            return
        
        log = lib.logger.Logger(self.stdout, self.verbose)
        
        ##### build cutoff info #####   
        
        self.distance_matrix   = ISDF_Local_Utils.get_cell_distance_matrix(self.cell)
        weight                 = np.sqrt(self.cell.vol / self.coords.shape[0])
        precision              = self.aoR_cutoff
        rcut                   = ISDF_Local_Utils._estimate_rcut(self.cell, self.coords.shape[0], precision)
        rcut_max               = np.max(rcut)
        atm2_bas               = ISDF_Local_Utils._atm_to_bas(self.cell)
        self.AtmConnectionInfo = []
        
        for i in range(self.cell.natm):
            tmp = ISDF_Local_Utils.AtmConnectionInfo(self.cell, i, self.distance_matrix, precision, rcut, rcut_max, atm2_bas)
            self.AtmConnectionInfo.append(tmp)
    
        #### information dealing grids , build parition ####
                
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                
        if Ls is None:
            Ls = [
                int(self.cell.lattice_vectors()[0][0]/2)+1,
                int(self.cell.lattice_vectors()[1][1]/2)+1,
                int(self.cell.lattice_vectors()[2][2]/2)+1
            ]
        
        self.partition_prim = ISDF_Local_Utils.get_partition(
            self.cell, self.coords,
            self.AtmConnectionInfo,
            Ls,
            self.with_translation_symmetry, 
            self.kmesh,
            self.use_mpi
        ) ## the id of grid points of self.partition_prim is w.r.t the supercell ##
        
        for i in range(len(self.partition_prim)):
            self.partition_prim[i] = np.array(self.partition_prim[i], dtype=np.int32)
        
        assert len(self.partition_prim) == self.natmPrim ## the grid id is the global grid id 
        
        self.partition = _expand_partition_prim(self.partition_prim, self.kmesh, self.mesh)
        
        assert len(self.partition) == self.natm
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        #### 
        
        if self.verbose:
            _benchmark_time(t1, t2, "build_partition", self)
        
        #### build aoR #### 
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        sync_aoR = False
        if self.direct:
            sync_aoR = True
        
        ## deal with translation symmetry ##
        
        first_natm = self.first_natm
        natm       = self.cell.natm
        
        #################################### 
        
        sync_aoR = False
        if self.direct:
            sync_aoR = True
        
        ### we need three types of aoR ### 
        
        # this type of aoR is used in get J and select IP 
        
        weight = np.sqrt(self.cell.vol / self.coords.shape[0])
        
        self.aoR = ISDF_Local_Utils.get_aoR(self.cell, self.coords, self.partition, 
                                                  first_natm,
                                                  natm,
                                                  self.group_global,
                                                  self.distance_matrix, 
                                                  self.AtmConnectionInfo, 
                                                  self.use_mpi, self.use_mpi, sync_aoR)
        
    
        memory = ISDF_Local_Utils._get_aoR_holders_memory(self.aoR) ### full col
        assert len(self.aoR) == first_natm
        # print("In ISDF-K build_partition_aoR aoR memory: %d " % (memory))
        log.info("In ISDF-K build_partition_aoR aoR memory: %d " % (memory))
        
        # if rank == 0:
        #     print("aoR memory: ", memory) 
        
        weight = np.sqrt(self.cell.vol / self.coords.shape[0])
        self.aoR1 = ISDF_Local_Utils.get_aoR(self.cell, self.coords, self.partition, 
                                                   None,
                                                   first_natm,
                                                   self.group_global,
                                                   self.distance_matrix, 
                                                   self.AtmConnectionInfo, 
                                                   self.use_mpi, self.use_mpi, sync_aoR)
        
        memory = ISDF_Local_Utils._get_aoR_holders_memory(self.aoR1)  ### full row 
        assert len(self.aoR1) == natm
        log.info("In ISDF-K build_partition_aoR aoR1 memory: %s", memory)
        partition_activated = None
        
        ##### the following info is used in get_J ##### 
        
        if not self.use_mpi:
            rank = 0
        if rank == 0:
            partition_activated = []
            for _id_, aoR_holder in enumerate(self.aoR1):
                if aoR_holder.ao_involved.size == 0:
                    partition_activated.append(False)
                else:
                    partition_activated.append(True)
            partition_activated = np.array(partition_activated, dtype=bool)
        if self.use_mpi:
            partition_activated = bcast(partition_activated)
        self.partition_activated = partition_activated
        self.partition_activated_id = []
        for i in range(len(partition_activated)):
            if partition_activated[i]:
                self.partition_activated_id.append(i)
        self.partition_activated_id = np.array(self.partition_activated_id, dtype=np.int32)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if self.verbose:
            _benchmark_time(t1, t2, "build_aoR", self)
    
    def build_IP_local(self, c=5, m=5, first_natm=None, group=None, Ls = None, debug=True):
        
        assert self.use_aft_ao == False
        
        first_natm = self.first_natm 
        if group is None:
            group = []
            for i in range(first_natm):
                group.append([i])
        
        ## check the group ##
        
        natm_involved = 0
        for data in group:
            for atm_id in data:
                assert atm_id < first_natm
            natm_involved += len(data)
        assert natm_involved == first_natm 
    
        for i in range(len(group)):
            group[i] = np.array(group[i], dtype=np.int32)
        
        assert len(group) <= first_natm
        
        # build partition and aoR # 
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        self.group = group
        
        self.group_global = []
        shift = 0
        self.atm_ordering = []
        for ix in range(self.kmesh[0]):
            for iy in range(self.kmesh[1]):
                for iz in range(self.kmesh[2]):
                    for data in self.group:
                        self.group_global.append(data + shift)
                        self.atm_ordering.extend(data + shift)
                    shift += self.natmPrim
        self.atm_ordering = np.array(self.atm_ordering, dtype=np.int32)
                
        self.atm_id_2_group = np.zeros((self.cell.natm), dtype=np.int32)
        for i in range(len(self.group_global)):
            for atm_id in self.group_global[i]:
                self.atm_id_2_group[atm_id] = i
        
        self.build_partition_aoR(None)
        
        self.grid_segment = [0]
        for atm_id in self.atm_ordering:
            # print("self.partition[atm_id] = ", self.partition[atm_id])
            loc_now = self.grid_segment[-1] + len(self.partition[atm_id])
            self.grid_segment.append(loc_now)
            # self.grid_segment.append(self.grid_segment[-1] + len(self.partition[atm_id]))
        self.grid_segment = np.array(self.grid_segment, dtype=np.int32)
        #print("grid_segment = ", self.grid_segment)
        
        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao
        
        self.partition_atmID_to_gridID = partition
        
        self.partition_group_to_gridID = []
        for i in range(len(group)):
            self.partition_group_to_gridID.append([])
            for atm_id in group[i]:
                self.partition_group_to_gridID[i].extend(partition[atm_id])
            self.partition_group_to_gridID[i] = np.array(self.partition_group_to_gridID[i], dtype=np.int32)
            # self.partition_group_to_gridID[i].sort()
            
        ngrids = self.coords.shape[0]
        
        gridID_2_atmID = np.zeros((ngrids), dtype=np.int32)
        
        for atm_id in range(self.cell.natm):
            gridID_2_atmID[partition[atm_id]] = atm_id
        
        self.gridID_2_atmID = gridID_2_atmID
        self.grid_ID_ordered = _get_grid_ordering_k(self.partition_group_to_gridID, self.kmesh, self.mesh)
        
        self.grid_ID_ordered_prim = self.grid_ID_ordered[:ngrids//np.prod(self.kmesh)].copy()
        
        self.partition_group_to_gridID = _expand_partition_prim(self.partition_group_to_gridID, self.kmesh, self.mesh)
        
        for i in range(len(self.grid_ID_ordered_prim)):
            grid_ID = self.grid_ID_ordered_prim[i]
            
            ix = grid_ID // (self.mesh[1] * self.mesh[2])
            iy = (grid_ID % (self.mesh[1] * self.mesh[2])) // self.mesh[2]
            iz = grid_ID % self.mesh[2]
            
            # assert ix < self.meshPrim[0]
            # assert iy < self.meshPrim[1]
            # assert iz < self.meshPrim[2]
            
            self.grid_ID_ordered_prim[i] = ix * self.meshPrim[1] * self.meshPrim[2] + iy * self.meshPrim[2] + iz
            
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "build_partition_aoR", self)
        
        t1 = t2 
        
        if len(group) < first_natm:
            IP_Atm = ISDF_LinearScaling.select_IP_atm_ls(
                self, c+1, m, first_natm, 
                rela_cutoff=self.rela_cutoff_QRCP,
                no_retriction_on_nIP=self.no_restriction_on_nIP,
                use_mpi=self.use_mpi
            )
        else:
            IP_Atm = ISDF_LinearScaling.select_IP_atm_ls(
                self, c, m, first_natm, 
                rela_cutoff=self.rela_cutoff_QRCP,
                no_retriction_on_nIP=self.no_restriction_on_nIP,
                use_mpi=self.use_mpi
            )
        
        t3 = (lib.logger.process_clock(), lib.logger.perf_counter()) 
        
        weight = np.sqrt(self.cell.vol / self.coords.shape[0])
        
        self.aoRg_possible = ISDF_Local_Utils.get_aoR(
            self.cell, self.coords, 
            IP_Atm, 
            first_natm,
            natm,
            self.group,
            self.distance_matrix, 
            self.AtmConnectionInfo, 
            self.use_mpi, self.use_mpi, True
        )
        
        assert len(self.aoRg_possible) == first_natm
        
        t4 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if self.verbose and debug:
            _benchmark_time(t3, t4, "build_aoRg_possible", self)
        
        build_aoR_FFT = (self.direct == False)
        
        select_IP_local_ls_k_drive(
            self, c, m, IP_Atm, group, 
            build_aoR_FFT=build_aoR_FFT,
            use_mpi=self.use_mpi
        )
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "select_IP", self)
        
        t1 = t2 
        
        ISDF_LinearScaling.build_aux_basis_ls(
            self, group, self.IP_group, debug=debug, use_mpi=self.use_mpi
        )
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "build_aux_basis", self)
    
        t1 = t2
        
        #self.aoR_Full = []
        ##self.aoRg_FUll = []
        
        #for i in range(self.kmesh[0]):
        #    for j in range(self.kmesh[1]):
        #        for k in range(self.kmesh[2]):
        #            self.aoR_Full.append(self._get_aoR_Row(i, j, k))   # different row -> different box -> a column permutation of the first box
        #            #self.aoRg_FUll.append(self._get_aoRg_Row(i, j, k))
        
        
        sys.stdout.flush()
    
    def build_auxiliary_Coulomb(self, debug=True):
        
        if self.direct == False:
            build_auxiliary_Coulomb_local_bas_k(self, debug=debug, use_mpi=self.use_mpi)

        ################ allocate buffer ################ 
    
    def _get_bufsize_get_j(self):
        
        # if self.with_robust_fitting == False:
        if True:
            
            naux       = self.naux
            nao        = self.nao
            nIP_Prim   = self.nIP_Prim
            nao_prim   = self.nao // np.prod(self.kmesh)
            
            size_buf3  = nao * naux + naux + naux + nao * nao
            size_buf4  = nao * nIP_Prim
            size_buf4 += nIP_Prim
            size_buf4 += nao_prim * nao
            size_buf4 += nIP_Prim
            size_buf4 += nao_prim * nao_prim
            size_buf4 += nao_prim * nIP_Prim * 3
            
            return max(size_buf3, size_buf4)
            
        # else:
        #     raise NotImplementedError

    def _get_bufsize_get_k(self):
        
        # if self.with_robust_fitting == False:
        if self.with_robust_fitting == False:
            
            naux     = self.naux
            nao      = self.nao
            nIP_Prim = self.nIP_Prim
            nao_prim = self.nao // np.prod(self.kmesh)
            ncell_complex = self.kmesh[0] * self.kmesh[1] * (self.kmesh[2]//2+1)
            
            #### size of density matrix ####
            
            size_dm = nao_prim * nao_prim * ncell_complex * 2
            size_dm += nIP_Prim * nIP_Prim * ncell_complex * 2
            
            #### size of buf to construct dm ####
            
            size_buf5 = nao_prim * nao_prim * 2 * 2
            size_buf5 += nao_prim * nIP_Prim * 2 * 2
            
            size_fft_buf = nIP_Prim * nIP_Prim * ncell_complex * 2
            
            #### size of buf to construct K ####
            
            size_buf6  = nao_prim * nao_prim * ncell_complex * 2 # k-buf
            size_buf6 += nIP_Prim * nIP_Prim * 2     # buf_A
            size_buf6 += nao_prim * nIP_Prim * 2 *2  # buf_B/C
            size_buf6 += nao_prim * nao_prim * 2     # buf_D
        
            return size_dm + max(size_buf5, size_buf6, size_fft_buf)
        
        else:
            
            naux     = self.naux
            nao      = self.nao
            nIP_Prim = self.nIP_Prim
            nGrid_Prim = self.nGridPrim
            nao_prim = self.nao // np.prod(self.kmesh)
            ncell_complex = self.kmesh[0] * self.kmesh[1] * (self.kmesh[2]//2+1)
            
            #### size of density matrix ####
            
            size_dm = nao_prim * nao_prim * ncell_complex * 2
            size_dm += nIP_Prim * nGrid_Prim * ncell_complex * 2
            
            #### size of buf to construct dm ####
            
            size_buf5 = nao_prim * nao_prim * 2 
            size_buf5 += nao_prim * nIP_Prim * 2 
            size_buf5 += nao_prim * nGrid_Prim * 2 * 2
            size_buf5 += nIP_Prim * nGrid_Prim * 2 
            
            size_fft_buf = nIP_Prim * nGrid_Prim * ncell_complex * 2
            
            #### size of buf to construct K ####
            
            size_buf6  = nao_prim * nao_prim * ncell_complex * 2 # k-buf
            size_buf6 += nIP_Prim * nGrid_Prim * 2     # buf_A
            size_buf6 += nao_prim * nGrid_Prim * 2     # buf_B
            size_buf6 += nao_prim * nIP_Prim * 2 * 2   # buf_B2/C
            size_buf6 += nao_prim * nao_prim * 2       # buf_D
        
            return size_dm + max(size_buf5, size_buf6, size_fft_buf)

    def _allocate_jk_buffer(self, dtype=np.float64):
        
        if self.jk_buffer is not None:
            return
            
        num_threads = lib.num_threads()
        
        nIP_Prim = self.nIP_Prim
        nGridPrim = self.nGridPrim
        ncell_complex = self.kmesh[0] * self.kmesh[1] * (self.kmesh[2]//2+1)
        nao_prim  = self.nao // np.prod(self.kmesh)
        naux       = self.naux
        nao        = self.nao
        ngrids = nGridPrim * self.kmesh[0] * self.kmesh[1] * self.kmesh[2]
        ncell  = np.prod(self.kmesh)
        
        self.outcore = False 
        
        if self.outcore is False:
            
            ### in build aux basis ###
            
            size_buf1 = nIP_Prim * ncell_complex*nIP_Prim * 2
            size_buf1+= nIP_Prim * ncell_complex*nGridPrim * 2 * 2
            size_buf1+= num_threads * nGridPrim * 2
            size_buf1+= nIP_Prim * nIP_Prim * 2
            size_buf1+= nIP_Prim * nGridPrim * 2 * 2
            size_buf1 = 0
            
            ### in construct W ###
            
            # print("nIP_Prim = ", nIP_Prim)
            # print("ncell_complex = ", ncell_complex)    
            
            size_buf2  = nIP_Prim * nIP_Prim * 2
            size_buf2 += nIP_Prim * nGridPrim * 2 * 2
            size_buf2 += nIP_Prim * nIP_Prim *  ncell_complex * 2 * 2
            size_buf2 = 0
            
            # print("size_buf2 = ", size_buf2)
            
            ### in get_j ###
                    
            buf_J = self._get_bufsize_get_j()
            buf_J = 0
            
            ### in get_k ### 
        
            buf_K = self._get_bufsize_get_k()
            
            ### ddot_buf ###
            
            # size_ddot_buf = max(naux*naux+2,ngrids)*num_threads
            size_ddot_buf = (nIP_Prim*nIP_Prim+2)*num_threads
            
            # print("size_buf1 = ", size_buf1)
            # print("size_buf2 = ", size_buf2)
            # print("size_buf3 = ", size_buf3)
            # print("size_buf4 = ", size_buf4)
            # print("size_buf5 = ", size_buf5)
            
            size_buf = max(size_buf1,size_buf2,buf_J,buf_K)
            
            # print("size_buf = ", size_buf)
            
            if hasattr(self, "IO_buf"):
                if self.IO_buf.size < (size_buf+size_ddot_buf):
                    self.IO_buf = np.zeros((size_buf+size_ddot_buf), dtype=np.float64)
                self.jk_buffer = np.ndarray((size_buf), dtype=np.float64, buffer=self.IO_buf, offset=0)
                self.ddot_buf  = np.ndarray((size_ddot_buf), dtype=np.float64, buffer=self.IO_buf, offset=size_buf)

            else:

                self.jk_buffer = np.ndarray((size_buf), dtype=np.float64)
                self.ddot_buf  = np.zeros((size_ddot_buf), dtype=np.float64)

    ##### all the following functions are used to deal with translation symmetry when getting j and getting k #####
    
    def _get_permutation_column_aoR(self, box_x, box_y, box_z, loc_internal=None):
        
        assert box_x < self.kmesh[0]
        assert box_y < self.kmesh[1]
        assert box_z < self.kmesh[2]
        
        if hasattr(self, "aoR_col_permutation") is False:
            self.aoR_col_permutation = []
            for i in range(np.prod(self.kmesh)):
                self.aoR_col_permutation.append(None)
        
        loc = box_x * self.kmesh[1] * self.kmesh[2] + box_y * self.kmesh[2] + box_z 
        
        if self.aoR_col_permutation[loc] is None:
            ### construct the permutation matrix ###
            permutation = []
            for aoR_holder in self.aoR:
                ao_involved = aoR_holder.ao_involved
                ao_permutated = []
                for ao_id in ao_involved:
                    box_id = ao_id // self.nao_prim
                    nao_id = ao_id % self.nao_prim
                    box_x_ = box_id // (self.kmesh[1] * self.kmesh[2])
                    box_y_ = (box_id % (self.kmesh[1] * self.kmesh[2])) // self.kmesh[2]
                    box_z_ = box_id % self.kmesh[2]
                    box_x_new = (box_x + box_x_) % self.kmesh[0]
                    box_y_new = (box_y + box_y_) % self.kmesh[1]
                    box_z_new = (box_z + box_z_) % self.kmesh[2]
                    nao_id_new = box_x_new * self.kmesh[1] * self.kmesh[2] * self.nao_prim + box_y_new * self.kmesh[2] * self.nao_prim + box_z_new * self.nao_prim + nao_id
                    ao_permutated.append(nao_id_new)
                # print("ao_permutated = ", ao_permutated)
                permutation.append(np.array(ao_permutated, dtype=np.int32))
            self.aoR_col_permutation[loc] = permutation
        
        if loc_internal is not None:
            return self.aoR_col_permutation[loc][loc_internal]
        else:
            return self.aoR_col_permutation[loc]
    
    def _get_permutation_column_aoRg(self, box_x, box_y, box_z, loc_internal=None):
        
        assert box_x < self.kmesh[0]
        assert box_y < self.kmesh[1]
        assert box_z < self.kmesh[2]
    
        if hasattr(self, "aoRg_col_permutation") is False:
            self.aoRg_col_permutation = []
            for i in range(np.prod(self.kmesh)):
                self.aoRg_col_permutation.append(None)
        
        loc = box_x * self.kmesh[1] * self.kmesh[2] + box_y * self.kmesh[2] + box_z
        
        if self.aoRg_col_permutation[loc] is None:
            ### construct the permutation matrix ###
            permutation = []
            for aoRg_holder in self.aoRg:
                ao_involved = aoRg_holder.ao_involved
                ao_permutated = []
                for ao_id in ao_involved:
                    box_id = ao_id // self.nao_prim
                    nao_id = ao_id % self.nao_prim
                    box_x_ = box_id // (self.kmesh[1] * self.kmesh[2])
                    box_y_ = (box_id % (self.kmesh[1] * self.kmesh[2])) // self.kmesh[2]
                    box_z_ = box_id % self.kmesh[2]
                    box_x_new = (box_x + box_x_) % self.kmesh[0]
                    box_y_new = (box_y + box_y_) % self.kmesh[1]
                    box_z_new = (box_z + box_z_) % self.kmesh[2]
                    nao_id_new = box_x_new * self.kmesh[1] * self.kmesh[2] * self.nao_prim + box_y_new * self.kmesh[2] * self.nao_prim + box_z_new * self.nao_prim + nao_id
                    ao_permutated.append(nao_id_new)
                permutation.append(np.array(ao_permutated, dtype=np.int32))
            self.aoRg_col_permutation[loc] = permutation
        
        if loc_internal is not None:
            return self.aoRg_col_permutation[loc][loc_internal]
        else:
            return self.aoRg_col_permutation[loc]
    
    # def get_aoR_Row(self, box_x, box_y, box_z):
    #     loc = box_x * self.kmesh[1] * self.kmesh[2] + box_y * self.kmesh[2] + box_z
    #     return self.aoR_Full[loc]
    
    # def _get_aoR_Row(self, box_x, box_y, box_z):
    #     assert box_x < self.kmesh[0]
    #     assert box_y < self.kmesh[1]
    #     assert box_z < self.kmesh[2]
    #     if box_x == 0 and box_y == 0 and box_z == 0:
    #         return self.aoR1
    #     else:
    #         Res = []
    #         for ix in range(self.kmesh[0]):
    #             for iy in range(self.kmesh[1]):
    #                 for iz in range(self.kmesh[2]):
    #                     ix_ = (ix - box_x + self.kmesh[0]) % self.kmesh[0]
    #                     iy_ = (iy - box_y + self.kmesh[1]) % self.kmesh[1]
    #                     iz_ = (iz - box_z + self.kmesh[2]) % self.kmesh[2]
    #                     loc_ = ix_ * self.kmesh[1] * self.kmesh[2] + iy_ * self.kmesh[2] + iz_
    #                     for i in range(loc_*self.natmPrim, (loc_+1)*self.natmPrim):
    #                         Res.append(self.aoR1[i])
    #         return Res
    
    def _get_aoRg_Row(self, box_x, box_y, box_z):
        
        assert box_x < self.kmesh[0]
        assert box_y < self.kmesh[1]
        assert box_z < self.kmesh[2]
        
        if box_x == 0 and box_y == 0 and box_z == 0:
            return self.aoRg1
        else:
            Res = []
            for ix in range(self.kmesh[0]):
                for iy in range(self.kmesh[1]):
                    for iz in range(self.kmesh[2]):
                        ix_ = (ix - box_x + self.kmesh[0]) % self.kmesh[0]
                        iy_ = (iy - box_y + self.kmesh[1]) % self.kmesh[1]
                        iz_ = (iz - box_z + self.kmesh[2]) % self.kmesh[2]
                        loc_ = ix_ * self.kmesh[1] * self.kmesh[2] + iy_ * self.kmesh[2] + iz_
                        for i in range(loc_*self.natmPrim, (loc_+1)*self.natmPrim):
                            Res.append(self.aoRg1[i])
            return Res

                       
    # get_jk = get_jk_dm_translation_symmetry

    #### subroutine to deal with _ewald_exxdiv_for_G0

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError
            # with self.range_coulomb(omega) as rsh_df:
            #     return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
            #                          omega=None, exxdiv=exxdiv)
        
        from pyscf.pbc.df.aft import _check_kpts
        
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            assert np.allclose(kpts[0], np.zeros(3))
            vj, vk = get_jk_dm_translation_symmetry(self, dm, hermi, kpts[0], kpts_band,
                                                    with_j, with_k, exxdiv=exxdiv)
        else:
            
            ### first construct J and K ### 
            
            from pyscf.pbc.df.isdf.isdf_linear_scaling_k_jk import _contract_j_dm_k_ls, _get_k_kSym_robust_fitting_fast, _get_k_kSym, _get_k_kSym_direct
            from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_dms, _format_kpts_band, _format_jks
            
            ### preprocess dm ### 
            
            if dm.ndim == 3:
                dm = dm.reshape(1, *dm.shape)
            nset = dm.shape[0]
            vj = np.zeros_like(dm, dtype=np.complex128)
            vk = np.zeros_like(dm, dtype=np.complex128)
            
            for iset in range(nset):
                vj[iset] = _contract_j_dm_k_ls(self, dm[iset])
                if self.with_robust_fitting:
                    if self.direct:
                        vk[iset] = _get_k_kSym_direct(self, dm[iset])
                    else:
                        vk[iset] = _get_k_kSym_robust_fitting_fast(self, dm[iset])
                else:
                    vk[iset] = _get_k_kSym(self, dm[iset])
            
            ### post process J and K ###
            
            kpts = np.asarray(kpts)
            dm_kpts = lib.asarray(dm, order='C')
            assert dm_kpts.ndim == 4
            assert dm_kpts.shape[1] == len(kpts)
            assert dm_kpts.shape[2] == dm_kpts.shape[3]
            dms = _format_dms(dm_kpts, kpts)
            nset, nkpts, nao = dms.shape[:3]
            assert nset <= 2
                        
            kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
            nband = len(kpts_band)
            assert nband == nkpts
            
            vk_kpts = vk.reshape(nset, nband, nao, nao)
            
            cell = self.prim_cell
            
            if exxdiv == 'ewald':
                _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)
            
            vk = _format_jks(vk_kpts, dm_kpts, input_band, kpts)
            vj_kpts = vj.reshape(nset, nband, nao, nao)
            vj = _format_jks(vj_kpts, dm_kpts, input_band, kpts)
            
            if nset == 1:
                vj = vj[0]
                vk = vk[0]
            
        return vj, vk

if __name__ == "__main__":

    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
    C = 25
    
    verbose = 10
    import pyscf.pbc.gto as pbcgto
    
    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
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
    
    KE_CUTOFF = 70
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    # prim_partition = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # prim_partition = [[0,1,2,3,4,5,6,7]]
    prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    
    Ls = [2, 2, 2]
    kpts = prim_cell.make_kpts(Ls)
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     #basis=basis, pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    
    # pbc_isdf_info = PBC_ISDF_Info_Quad_K(cell, kmesh=Ls, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, rela_cutoff_QRCP=3e-3)
    pbc_isdf_info = PBC_ISDF_Info_Quad_K(prim_cell, kmesh=Ls, with_robust_fitting=True, aoR_cutoff=1e-8, direct=True, rela_cutoff_QRCP=3e-3,
                                         limited_memory=True, build_K_bunchsize=32)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=prim_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    pbc_isdf_info.verbose = 10
    
    # exit(1)
    
    #print("grid_segment         = ", pbc_isdf_info.grid_segment)
    #print("len of grid_ordering = ", len(pbc_isdf_info.grid_ID_ordered))
    
    #aoR_unpacked = []
    #for aoR_holder in pbc_isdf_info.aoR1:
    #    aoR_unpacked.append(aoR_holder.todense(prim_cell.nao_nr()))
    #aoR_unpacked = np.concatenate(aoR_unpacked, axis=1)
    #print("aoR_unpacked shape = ", aoR_unpacked.shape)
    
    weight = np.sqrt(cell.vol / pbc_isdf_info.coords.shape[0])
    aoR_benchmark = ISDF_eval_gto(cell, coords=pbc_isdf_info.coords[pbc_isdf_info.grid_ID_ordered]) * weight
    
    # loc = 0
    # nao_prim = prim_cell.nao_nr()
    # for ix in range(Ls[0]):
    #     for iy in range(Ls[1]):
    #         for iz in range(Ls[2]):
    #             aoR_unpacked = []
    #             aoR_holder = pbc_isdf_info.get_aoR_Row(ix, iy, iz)
    #             for data in aoR_holder:
    #                 # print("data = ", data.aoR.shape)
    #                 aoR_unpacked.append(data.todense(nao_prim))
    #             aoR_unpacked = np.concatenate(aoR_unpacked, axis=1)
    #             aoR_benchmark_now = aoR_benchmark[loc*nao_prim:(loc+1)*nao_prim,:]
    #             loc += 1
    #             diff = aoR_benchmark_now - aoR_unpacked
    #             where = np.where(np.abs(diff) > 1e-4)
    #             print("diff = ", np.linalg.norm(diff)/np.sqrt(aoR_unpacked.size))
    # print("prim_mesh = ", prim_mesh)
    
    # exit(1)
    
    naux_prim = 0
    for data in pbc_isdf_info.aoRg:
        naux_prim += data.aoR.shape[1]
    print("naux_prim = ", naux_prim)
    print("naux = ", pbc_isdf_info.naux)
    
    aoR_unpacked = np.zeros_like(aoR_benchmark)
    ngrid = 0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                perm_col = pbc_isdf_info._get_permutation_column_aoR(ix, iy, iz)
                for _loc_, data in enumerate(pbc_isdf_info.aoR):
                    aoR_unpacked[perm_col[_loc_], ngrid:ngrid+data.aoR.shape[1]] = data.aoR
                    ngrid += data.aoR.shape[1]
    assert ngrid == np.prod(mesh)
    diff = aoR_benchmark - aoR_unpacked
    where = np.where(np.abs(diff) > 1e-4)
    print("where = ", where)
    print("diff = ", np.linalg.norm(diff)/np.sqrt(aoR_unpacked.size)) 
    
    ngrid_prim = np.prod(prim_mesh)
    diff = aoR_benchmark[:, :ngrid_prim] - aoR_unpacked[:,:ngrid_prim]
    print("diff.shape = ", diff.shape)
    print("diff = ", np.linalg.norm(diff)/np.sqrt(diff.size))
    where = np.where(np.abs(diff) > 1e-4)
    print("where = ", where)
    
    grid_ID_prim = pbc_isdf_info.grid_ID_ordered[:ngrid_prim]
    grid_ID_prim2 = []
    for i in range(pbc_isdf_info.natmPrim):
        grid_ID_prim2.extend(pbc_isdf_info.partition[i])
    grid_ID_prim2 = np.array(grid_ID_prim2, dtype=np.int32)
    assert np.allclose(grid_ID_prim, grid_ID_prim2)
    
    # exit(1)
    
    # pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    # print("grid_segment = ", pbc_isdf_info.grid_segment)
    
    from pyscf.pbc import scf

    mf = scf.KRHF(prim_cell, kpts)
    # pbc_isdf_info.kpts = np.array([[0,0,0]])  
    # mf = scf.addons.smearing_(mf, sigma=0.2, method='fermi')
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 1e-7
    
    mf.kernel()
    
    # exit(1)
    
    ######### bench mark #########
    
    pbc_isdf_info = ISDF_LinearScaling.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=True, rela_cutoff_QRCP=1e-3, use_occ_RI_K=False)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*3, Ls[1]*3, Ls[2]*3])
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    aoR_unpacked = []
    for aoR_holder in pbc_isdf_info.aoR:
        aoR_unpacked.append(aoR_holder.todense(cell.nao_nr()))
    aoR_unpacked = np.concatenate(aoR_unpacked, axis=1)
    grid_ordered = pbc_isdf_info.grid_ID_ordered
    aoR_benchmark = ISDF_eval_gto(cell, coords=pbc_isdf_info.coords[grid_ordered]) * weight
    diff = aoR_benchmark - aoR_unpacked
    print("diff = ", np.linalg.norm(diff)/np.sqrt(aoR_unpacked.size))
    # exit(1)
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 1e-7
    mf.kernel()
    
    # pp = mf.with_df.get_pp()
    # mf = scf.RHF(cell)
    # pbc_isdf_info.direct_scf = mf.direct_scf
    # mf.with_df.get_pp = lambda *args, **kwargs: pp
    # mf.max_cycle = 16
    # mf.conv_tol = 1e-7
    # mf.kernel()