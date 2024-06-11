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

import numpy
import pyscf
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
import pyscf.pbc.gto as pbcgto

import numpy as np
import ctypes
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
from pyscf.pbc.df.isdf.isdf_ao2mo import LS_THC, LS_THC_eri, laplace_holder

##### deal with CC #####

from pyscf.cc.rccsd import _ChemistsERIs, RCCSD 

def _make_isdf_eris_incore(mycc, my_isdf:ISDF.PBC_ISDF_Info_Quad, mo_coeff=None):
    
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    eri1 = my_isdf.ao2mo(mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    
    cput1 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(cput0, cput1, "CCSD integral transformation")
    
    return eris

def RCCSD_isdf(mf, frozen=0, mo_coeff=None, mo_occ=None, run=True):
    mycc = RCCSD(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    # eris = mycc.ao2mo(mo_coeff)
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    eris_ccsd = _make_isdf_eris_incore(mycc, mf.with_df, mo_coeff=mo_coeff)
    # mycc.eris = eris
    if run:
        mycc.kernel(eris=eris_ccsd)
    return mycc, eris_ccsd

from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast

class _restricted_THC_posthf_holder:
    def __init__(self, my_isdf, my_mf, X,
                 laplace_rela_err = 1e-7,
                 laplace_order    = 2):
        
        if rank == 0:
        
            print("THC posthf holder is initialized!")
        
            self.my_isdf = my_isdf
            self.my_mf   = my_mf
            self.X       = X
        
            self.mo_coeff = my_mf.mo_coeff
            self.mo_occ   = my_mf.mo_occ
            self.mo_energy = my_mf.mo_energy
        
            self.nocc = my_mf.mol.nelectron // 2
            self.nvir = my_mf.mol.nao - self.nocc
        
            self.occ_ene = self.mo_energy[:self.nocc]
            self.vir_ene = self.mo_energy[self.nocc:]
        
            #### construct Z matrix for XXZXX #### 
        
            t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
            Z = LS_THC(my_isdf, X)
            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
            _benchmark_time(t1, t2, "Z matrix construction for THC eri = XXZXX")

            #### construct laplace ####
        
            self._laplace = laplace_holder(self.mo_energy, self.nocc,
                                           rel_error = laplace_rela_err,
                                           order = laplace_order)

            self.X_o = lib.ddot(self.mo_coeff[:, :self.nocc].T, X)
            self.X_v = lib.ddot(self.mo_coeff[:, self.nocc:].T, X)
            self.Z = Z
            self.tau_o = self._laplace.laplace_occ
            self.tau_v = self._laplace.laplace_vir
        
            self.n_laplace = self.tau_o.shape[1]
            self.nthc_int = self.Z.shape[1]
        
            print("n_laplace = ", self.n_laplace)
            print("nthc_int  = ", self.nthc_int)
            print("nocc      = ", self.nocc)
            print("nvir      = ", self.nvir)
        
            print("THC posthf holder initialization is finished!")
        
        else:
            
            self.my_isdf  = None
            self.my_mf    = None
            self.X        = None
            self.mo_coeff = None
            self.mo_occ   = None
            self.mo_energy = None
            self.nocc = None
            self.nvir = None
            self.occ_ene = None
            self.vir_ene = None
            self._laplace = None
            self.X_o = None
            self.X_v = None
            self.Z = None
            self.tau_o = None
            self.tau_v = None
            self.n_laplace = None
            self.nthc_int = None
        
        #### sync ####
        
        if comm_size > 1:
            
            #self.my_isdf  = bcast(self.my_isdf, 0)
            #self.my_mf    = bcast(self.my_mf, 0)
            self.X        = bcast(self.X, 0)
            self.mo_coeff = bcast(self.mo_coeff, 0)
            self.mo_occ   = bcast(self.mo_occ, 0)
            self.mo_energy = bcast(self.mo_energy, 0)
            self.nocc = bcast(self.nocc, 0)
            self.nvir = bcast(self.nvir, 0)
            self.occ_ene = bcast(self.occ_ene, 0)
            self.vir_ene = bcast(self.vir_ene, 0)
            # self._laplace = bcast(self._laplace, 0)
            self.X_o = bcast(self.X_o, 0)
            self.X_v = bcast(self.X_v, 0)
            self.Z = bcast(self.Z, 0)
            self.tau_o = bcast(self.tau_o, 0)
            self.tau_v = bcast(self.tau_v, 0)
            self.n_laplace = bcast(self.n_laplace, 0)
            self.nthc_int = bcast(self.nthc_int, 0)

            comm.barrier()


if __name__ == '__main__':

    for c in [25]:
        for N in [1]:

            print("Testing c = ", c, "N = ", N, "...")

            cell   = pbcgto.Cell()
            boxlen = 3.5668
            cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
            
            cell.atom = [
                ['C', (0.     , 0.     , 0.    )],
                ['C', (0.8917 , 0.8917 , 0.8917)],
                ['C', (1.7834 , 1.7834 , 0.    )],
                ['C', (2.6751 , 2.6751 , 0.8917)],
                ['C', (1.7834 , 0.     , 1.7834)],
                ['C', (2.6751 , 0.8917 , 2.6751)],
                ['C', (0.     , 1.7834 , 1.7834)],
                ['C', (0.8917 , 2.6751 , 2.6751)],
            ] 

            cell.basis   = 'gth-szv'
            cell.pseudo  = 'gth-pade'
            cell.verbose = 4
            cell.ke_cutoff = 128
            cell.max_memory = 800  # 800 Mb
            cell.precision  = 1e-8  # integral precision
            cell.use_particle_mesh_ewald = True
            
            verbose = 4
            
            prim_cell = build_supercell(cell.atom, cell.a, Ls = [1,1,1], ke_cutoff=cell.ke_cutoff, basis=cell.basis, pseudo=cell.pseudo)   
            prim_partition = [[0,1,2,3], [4,5,6,7]]
            prim_mesh = prim_cell.mesh
            
            Ls = [1, 1, N]
            Ls = np.array(Ls, dtype=np.int32)
            mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
            mesh = np.array(mesh, dtype=np.int32)
                        
            cell, group_partition = build_supercell_with_partition(
                                    cell.atom, cell.a, mesh=mesh, 
                                    Ls=Ls,
                                    basis=cell.basis, 
                                    pseudo=cell.pseudo,
                                    partition=prim_partition, ke_cutoff=cell.ke_cutoff, verbose=verbose) 
        
            ####### bench mark MP2 ####### 
            
            import numpy
            from pyscf.pbc import gto, scf, mp

            mf = scf.RHF(cell)
            # mf.kernel()
            
            mypt = mp.RMP2(mf)
            # mypt.kernel()
            
            ####### isdf MP2 can perform directly! ####### 
            
            myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
            myisdf.build_IP_local(c=c, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
            myisdf.build_auxiliary_Coulomb(debug=True)
            
            mf_isdf = scf.RHF(cell)
            myisdf.direct_scf = mf_isdf.direct_scf
            mf_isdf.with_df = myisdf
            mf_isdf.max_cycle = 8
            mf_isdf.conv_tol = 1e-8
            mf_isdf.kernel()
            
            # print("mo_energy = ", mf.mo_energy)
            print("mo_energy = ", mf_isdf.mo_energy)
            
            isdf_pt = mp.RMP2(mf_isdf)
            isdf_pt.kernel()
            
            ######################## CCSD ########################
            
            ## benchmark ##
            
            mycc = pyscf.cc.CCSD(mf)
            # mycc.kernel()
            
            mycc_isdf, eris_ccsd = RCCSD_isdf(mf_isdf, run=False)
            mycc_isdf.kernel(eris=eris_ccsd)
            
            ####### THC-DF ####### 
            
            _myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
            _myisdf.build_IP_local(c=15, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
            R          = _myisdf.aoRg_full()
            Z          = LS_THC(myisdf, R)
            eri_LS_THC = LS_THC_eri(Z, R) 
            print("eri_LS_THC = ", eri_LS_THC[0,0,0,0])
            eri_benchmark = myisdf.get_eri(compact=False)
            print("eri_benchmark = ", eri_benchmark[0,0,0,0])
            diff          = np.linalg.norm(eri_LS_THC - eri_benchmark)
            print("diff = ", diff/np.sqrt(eri_benchmark.size))
            
            ####### LAPLACE #######
            
            mo_ene = mf_isdf.mo_energy
            nocc   = mf_isdf.mol.nelectron // 2
            nvir   = mf_isdf.mol.nao - nocc
            
            print("mo_ene = ", mo_ene)
            print("nocc   = ", nocc)
            
            laplace = laplace_holder(mo_ene, nocc)
            
            occ_ene = mo_ene[:nocc]
            vir_ene = mo_ene[nocc:]
            
            delta_full = laplace.delta_full
            
            delta_full_benchmark = np.zeros((nocc, nocc, nvir, nvir))
            
            for a in range(nvir):
                for b in range(nvir):
                    for i in range(nocc):
                        for j in range(nocc):
                            delta_full_benchmark[i,j,a,b] = 1.0/(vir_ene[a] + vir_ene[b] - occ_ene[i] - occ_ene[j])
            
            diff = delta_full - delta_full_benchmark
            rela_diff = diff/delta_full_benchmark
            print("max rela_diff = ", np.max(np.abs(rela_diff)))
            
            print("delta_full = ", delta_full[0,0,0,:])
            print("delta_full_benchmark = ", delta_full_benchmark[0,0,0,:])
            
            ##### thc holder ##### 
            
            thc_holder = _restricted_THC_posthf_holder(myisdf, mf_isdf, R)