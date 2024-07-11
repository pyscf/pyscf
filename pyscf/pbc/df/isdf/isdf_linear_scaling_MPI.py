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

import numpy as np

from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.gto.mole import *
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF_LinearScaling

class PBC_ISDF_Info_Quad_MPI(ISDF_LinearScaling.PBC_ISDF_Info_Quad):
    ''' Interpolative separable density fitting (ISDF) for periodic systems with MPI.
    
    The locality is explored! 
    
    k-point sampling is not currently supported!
    
    '''

    # Quad stands for quadratic scaling
    
    def __init__(self, mol:Cell, 
                 Ls=None,
                 verbose = 1,
                 rela_cutoff_QRCP = None,
                 aoR_cutoff = 1e-8,
                 ):
        
        super().__init__(mol, True, Ls, verbose, rela_cutoff_QRCP, aoR_cutoff, True, use_occ_RI_K=False)
        self.use_mpi = True
        assert self.use_aft_ao == False

if __name__ == '__main__':

    C = 15
    from pyscf.lib.parameters import BOHR
    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
    
    verbose = 4
    if rank != 0:
        verbose = 0
    
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
    ke_cutoff = 128 
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    KE_CUTOFF = 128
        
    prim_mesh = prim_cell.mesh    
    prim_partition = [[0], [1], [2], [3]]    
    
    Ls = [2, 2, 1]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    if rank == 0:
        print("group_partition = ", group_partition)
    
    pbc_isdf_info = PBC_ISDF_Info_Quad_MPI(cell, aoR_cutoff=1e-8, verbose=verbose)
    # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*3, Ls[1]*3, Ls[2]*3])
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition)
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    from pyscf.pbc import scf

    if comm_size > 1:
        comm.Barrier()

    mf = scf.RHF(cell)
    mf = scf.addons.smearing_(mf, sigma=0.2, method='fermi')
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 0.0
    
    dm = mf.init_guess_by_atom()
    
    if comm_size > 1:
        dm = bcast(dm, root=0)
    
    mf.kernel(dm)
    
    comm.Barrier()