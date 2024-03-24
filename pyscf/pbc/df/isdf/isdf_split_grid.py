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

import ctypes

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

########## WARNING: ABANDON, THIS IDEA DOES NOT WORK ! !!! ##########

############ select IP ############

def _select_IP_given_group(mydf, c:int, m:int, group=None, IP_possible = None, use_mpi=False):
    
    if group is None:
        raise ValueError("group must be specified")

    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())
        
    nthread = lib.num_threads()
    
    coords = mydf.coords
    
    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    
    fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    
    #### perform QRCP ####

    nao_group = 0
    for atm_id in group:
        shl_begin = mydf.shl_atm[atm_id][0]
        shl_end   = mydf.shl_atm[atm_id][1]
        nao_atm = mydf.aoloc_atm[shl_end] - mydf.aoloc_atm[shl_begin]
        nao_group += nao_atm

    ### do not deal with this problem right now
    # buf_size_per_thread = mydf.get_buffer_size_in_IP_selection_given_atm(c, m)
    # buf_size = buf_size_per_thread
    # buf = mydf.IO_buf
    # buf_tmp = np.ndarray((buf_size), dtype=buf.dtype, buffer=buf)
    # buf_tmp[:buf_size_per_thread] = 0.0 
    
    ##### random projection #####

    nao = mydf.nao
    
    aoR_atm = ISDF_eval_gto(mydf.cell, coords=coords[IP_possible]) * weight
    

    # print("nao_group = ", nao_group)
    # print("nao = ", nao)    
    # print("c = %d, m = %d" % (c, m))

    naux_now = int(np.sqrt(c*nao)) + m
    G1 = np.random.rand(nao, naux_now)
    G1, _ = numpy.linalg.qr(G1)
    G1 = G1.T
    
    G2 = np.random.rand(nao, naux_now)
    G2, _ = numpy.linalg.qr(G2)
    G2    = G2.T 
    # naux_now = nao
        
    aoR_atm1 = lib.ddot(G1, aoR_atm)
    
    naux_now1 = aoR_atm1.shape[0]
    aoR_atm2 = lib.ddot(G2, aoR_atm)
    naux_now2 = aoR_atm2.shape[0]
    
    naux2_now = naux_now1 * naux_now2
    
    R = np.ndarray((naux2_now, IP_possible.shape[0]), dtype=np.float64)

    aoPairBuffer = np.ndarray((naux2_now, IP_possible.shape[0]), dtype=np.float64)

    fn_ik_jk_ijk(aoR_atm1.ctypes.data_as(ctypes.c_void_p),
                 aoR_atm2.ctypes.data_as(ctypes.c_void_p),
                 aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_now1),
                 ctypes.c_int(naux_now2),
                 ctypes.c_int(IP_possible.shape[0]))

    aoR_atm1 = None
    aoR_atm2 = None

    max_rank  = min(naux2_now, IP_possible.shape[0], nao_group * c)  
    # print("naux2_now = %d, max_rank = %d" % (naux2_now, max_rank))
    # print("IP_possible.shape = ", IP_possible.shape)
    # print("nao_group = ", nao_group)
    # print("c = ", c)
    # print("nao_group * c = ", nao_group * c)
    
    npt_find = ctypes.c_int(0)
    pivot    = np.arange(IP_possible.shape[0], dtype=np.int32)

    thread_buffer = np.ndarray((nthread+1, IP_possible.shape[0]+1), dtype=np.float64)
    global_buffer = np.ndarray((1, IP_possible.shape[0]), dtype=np.float64)

    fn_colpivot_qr(aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(naux2_now),
                   ctypes.c_int(IP_possible.shape[0]),
                   ctypes.c_int(max_rank),
                   ctypes.c_double(1e-14),
                   pivot.ctypes.data_as(ctypes.c_void_p),
                   R.ctypes.data_as(ctypes.c_void_p),
                   ctypes.byref(npt_find),
                   thread_buffer.ctypes.data_as(ctypes.c_void_p),
                   global_buffer.ctypes.data_as(ctypes.c_void_p))
    npt_find = npt_find.value

    cutoff   = abs(R[npt_find-1, npt_find-1])
    # print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (IP_possible.shape[0], npt_find, cutoff))
    pivot = pivot[:npt_find]
    pivot.sort()
    results = list(IP_possible[pivot])
    
    
    ### clean up ###
    
    aoPairBuffer = None
    R = None
    pivot = None
    thread_buffer = None
    global_buffer = None
    
    return results

def select_IP_local_drive(mydf, c, m, IP_possible_group, group, use_mpi=False):
    
    IP_group  = []

    ######### allocate buffer #########

    natm = mydf.natm
    
    for i in range(len(group)):
        IP_group.append(None)

    for i in range(len(group)):
        IP_group[i] = _select_IP_given_group(mydf, c, m, group=group[i], IP_possible=IP_possible_group[i], use_mpi=use_mpi)

    mydf.IP_group = IP_group
    
    mydf.IP_flat = []
    mydf.IP_segment = [0]
    nIP_now = 0
    for x in IP_group:
        mydf.IP_flat.extend(x)
        nIP_now += len(x)
        mydf.IP_segment.append(nIP_now)
    mydf.IP_flat = np.array(mydf.IP_flat, dtype=np.int32)
    
    ### build ### 
    
    coords = mydf.coords
    weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    mydf.aoRg = ISDF_eval_gto(mydf.cell, coords=coords[mydf.IP_flat]) * weight
    # mydf.aoRg1 = np.zeros_like(mydf.aoRg2)
    mydf.naux = mydf.aoRg.shape[1]
    
    print("IP_segment = ", mydf.IP_segment)
    
    return IP_group

############ build aux bas ############

def build_aux_basis(mydf, group, IP_group, debug=True, use_mpi=False):

    natm = mydf.natm

    #aux_basis = []
    
    print("mydf.naux = ", mydf.naux)
    print("mydf.ngrids = ", mydf.ngrids)

    aux_basis = np.zeros((mydf.naux, mydf.ngrids), dtype=mydf.aoRg.dtype)

    aoRg = mydf.aoRg

    for i in range(len(group)):
        # ao_loc_begin = mydf.aoloc_atm[mydf.shl_atm[i][0]]
        # ao_loc_end   = mydf.aoloc_atm[mydf.shl_atm[i][1]]
        
        IP_loc_begin = mydf.IP_segment[i]
        IP_loc_end   = mydf.IP_segment[i+1]
        
        aoRg1 = aoRg[:,IP_loc_begin:IP_loc_end]
        
        A = lib.ddot(aoRg1.T, aoRg1)
        A = A * A 
        # B1 = lib.ddot(aoRg1.T, mydf.aoR[ao_loc_begin:ao_loc_end,:])
        grid_ID = mydf.partition_group_to_gridID[i]
        B = lib.ddot(aoRg1.T, mydf.aoR[:, grid_ID])
        B = B * B 
        
        with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
            e, h = scipy.linalg.eigh(A)
        
        print("condition number = ", e[-1]/e[0])
        where = np.where(e > e[-1]*1e-16)[0]
        e = e[where]
        h = h[:,where]
        
        B = lib.ddot(h.T, B)
        lib.d_i_ij_ij(1.0/e, B, out=B)
        aux_tmp = lib.ddot(h, B)
        # aux_basis.append(lib.ddot(h, B))
        aux_basis[IP_loc_begin:IP_loc_end, grid_ID] = aux_tmp
        
        e = None
        h = None
        B = None
        A1 = None
        A2 = None
        A = None
        B1 = None
        B2 = None
        aux_tmp = None
    
    mydf.aux_basis = aux_basis
    
    print("aux_basis.shape = ", mydf.aux_basis.shape)

class PBC_ISDF_Info_SplitGrid(ISDF.PBC_ISDF_Info):
    
    def __init__(self, mol:Cell, aoR: np.ndarray = None,
                 with_robust_fitting=True,
                 Ls=None,
                 get_partition=True,
                 verbose = 1
                 ):
    
        super().__init__(
            mol=mol,
            aoR=aoR,
            with_robust_fitting=with_robust_fitting,
            Ls=Ls,
            get_partition=get_partition,
            verbose=verbose
        )
        
        shl_atm = []
        
        natm = self.natm
        cell = self.cell
        
        for i in range(natm):
            shl_atm.append([None, None])
        
        for i in range(cell.nbas):
            atm_id = cell.bas_atom(i)
            if shl_atm[atm_id][0] is None:
                shl_atm[atm_id][0] = i
            shl_atm[atm_id][1] = i+1
        
        self.shl_atm = shl_atm
        self.aoloc_atm = cell.ao_loc_nr()

    def get_buffer_size_in_IP_selection_given_atm(self, c, m):
        pass

    def build_IP_Local(self, c=5, m=5,
                       # global_IP_selection=True,
                       build_global_basis=True,
                       IP_ID=None,
                       group=None,
                       debug=True):
    
        # build partition

        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao
        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao
        
        possible_IP = None
        if IP_ID is None:
            IP_ID = ISDF._select_IP_direct(self, c+1, m, global_IP_selection=False) # get a little bit more possible IPs
            IP_ID.sort()
            IP_ID = np.array(IP_ID, dtype=np.int32)
        possible_IP = np.array(IP_ID, dtype=np.int32)

        coords = self.coords
        weight = np.sqrt(self.cell.vol / coords.shape[0])
        aoR_possible_IP = ISDF_eval_gto(self.cell, coords=coords[possible_IP]) * weight
        
        self.aoR_possible_IP = aoR_possible_IP
        self.possible_IP = possible_IP
        
        if group==None:
            group = []
            for i in range(natm):
                group.append([i])
        
        possible_IP_atm = []
        for i in range(natm):
            possible_IP_atm.append([])
        for ip_id in possible_IP:
            atm_id = self.partition[ip_id]
            possible_IP_atm[atm_id].append(ip_id)
        for i in range(natm):
            possible_IP_atm[i] = np.array(possible_IP_atm[i], dtype=np.int32)
            possible_IP_atm[i].sort()
        
        possible_IP_group = []
        for i in range(len(group)):
            possible_IP_group.append([])
            for atm_id in group[i]:
                # print("atm_id = ", atm_id)
                # print("possible_IP_atm[atm_id] = ", possible_IP_atm[atm_id])
                possible_IP_group[i].extend(possible_IP_atm[atm_id])
            possible_IP_group[i].sort()
            possible_IP_group[i] = np.array(possible_IP_group[i], dtype=np.int32)
        
        partition_atmID_to_gridID = []
        for i in range(natm):
            partition_atmID_to_gridID.append([])
        for i in range(len(partition)):
            partition_atmID_to_gridID[partition[i]].append(i) # this can be extremely slow
        for i in range(natm):
            partition_atmID_to_gridID[i] = np.array(partition_atmID_to_gridID[i], dtype=np.int32)
            partition_atmID_to_gridID[i].sort()
        self.partition_atmID_to_gridID = partition_atmID_to_gridID
        self.partition_group_to_gridID = []
        for i in range(len(group)):
            self.partition_group_to_gridID.append([])
            for atm_id in group[i]:
                self.partition_group_to_gridID[i].extend(partition_atmID_to_gridID[atm_id])
            self.partition_group_to_gridID[i] = np.array(self.partition_group_to_gridID[i], dtype=np.int32)
            self.partition_group_to_gridID[i].sort()
        
        self.group = group
        select_IP_local_drive(self, c, m, possible_IP_group, group, use_mpi=False)
    
        build_aux_basis(self, group, self.IP_group, debug=True, use_mpi=False)
        
        self.naux = self.aux_basis.shape[0]
    
    def check_AOPairError(self):
        
        print("In check_AOPairError")
        
        for i in range(len(self.group)):
        # for i in range(1):
            
            IP_begin = self.IP_segment[i]
            IP_end   = self.IP_segment[i+1]
                
            print("group[%d] = " % i, self.group[i])
            print("IP_segment[%d] = %d, %d" % (i, IP_begin, IP_end))
                
            for atm_id in self.group[i]:
            
                ao_loc_begin = self.aoloc_atm[self.shl_atm[atm_id][0]]
                ao_loc_end   = self.aoloc_atm[self.shl_atm[atm_id][1]]
            
                aoRg1 = self.aoRg1[ao_loc_begin:ao_loc_end, IP_begin:IP_end]
                aoRg2 = self.aoRg2[:,IP_begin:IP_end]            
                coeff = numpy.einsum('ik,jk->ijk', aoRg1, aoRg2).reshape(-1, IP_end-IP_begin)        
                basis = self.aux_basis[IP_begin:IP_end,:]
                aux_approx = coeff @ basis
            
                # aoPair = numpy.einsum('ik,jk->ijk', self.aoR[ao_loc_begin:ao_loc_end, :], np.vstack([self.aoR[:ao_loc_begin, :], self.aoR[ao_loc_end:, :]])).reshape(-1, self.aoR.shape[1])
                aoPair = numpy.einsum('ik,jk->ijk', self.aoR[ao_loc_begin:ao_loc_end, :], self.aoR).reshape(-1, self.aoR.shape[1])
            
                diff = aux_approx - aoPair
            
                diff_pair_abs_max = np.max(np.abs(diff), axis=1)
            
                for j in range(ao_loc_end-ao_loc_begin):
                    loc_now = 0
                    for k in range(self.nao):
                        loc = j * self.nao + k
                        print("diff_pair_abs_max[%d,%d] = %12.6e" % (ao_loc_begin+j,k, diff_pair_abs_max[loc])) 
            
                # for i in range(diff_pair_abs_max.shape[0]):
                #     print("diff_pair_abs_max[%d] = %12.6e" % (i, diff_pair_abs_max[i]))

    get_jk = isdf_jk.get_jk_dm

C = 12

#### split over grid points ? ####

if __name__ == '__main__':

    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    cell.atom = '''
                   C     0.      0.      0.
                   C     0.8917  0.8917  0.8917
                   C     1.7834  1.7834  0.
                   C     2.6751  2.6751  0.8917
                   C     1.7834  0.      1.7834
                   C     2.6751  0.8917  2.6751
                   C     0.      1.7834  1.7834
                   C     0.8917  2.6751  2.6751
                '''
    
    cell.basis   = 'gth-dzvp'
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    # cell.ke_cutoff  = 128   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 70
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()
    
    cell = tools.super_cell(cell, [1, 1, 2])
    
    print(cell.ao_loc_nr())
    
    shl_atm = []
        
    natm = cell.natm
    # cell = self.cell
        
    for i in range(natm):
        shl_atm.append([None, None])
        
    for i in range(cell.nbas):
        atm_id = cell.bas_atom(i)
        if shl_atm[atm_id][0] is None:
            shl_atm[atm_id][0] = i
        shl_atm[atm_id][1] = i+1
    
    print("shl_atm = ", shl_atm) 
    
    ########## get aoR ##########
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    nx = grids.mesh[0]

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)
    
    ##############################
    
    pbc_isdf_info = PBC_ISDF_Info_SplitGrid(cell, aoR)
    # pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C, group=[[0,1,2,3,4,5,6,7]])
    # pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C, group=[[0,1],[2,3],[4,5],[6,7]])
    # pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C, group=[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]])
    # pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C)
    # pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C, group=[[0,1,2,3],[4,5,6,7]])
    pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C, group=[[0,1,2,3],[4,5,6,7], [8,9,10,11], [12,13,14,15]])
    print(pbc_isdf_info.IP_group) 
    
    # pbc_isdf_info.check_AOPairError()

    pbc_isdf_info.build_auxiliary_Coulomb()
    
    # exit(1)
    
    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7

    print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()