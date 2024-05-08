# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

# from pyscf.pbc.df.isdf import ISDF

from pyscf.pbc.df.isdf import isdf_linear_scaling

import numpy as np
from pyscf import ao2mo
from pyscf.pbc.df import aft
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc import tools
from pyscf.pbc import df


def _check_eri(eri, eri_bench, tol=1e-7):
    assert eri.shape == eri_bench.shape

    if len(eri.shape) == 2:
        for i in range(eri.shape[0]):
            for j in range(eri.shape[1]):
                if abs(eri_bench[i,j] - eri[i,j]) > tol:
                    print("eri[{}, {}] = {} != {}".format(i,j,eri_bench[i,j,], eri[i,j]),
                          "ration = ", eri_bench[i,j]/eri[i,j])
    else:
        assert len(eri.shape) == 4
        for i in range(eri.shape[0]):
            for j in range(eri.shape[1]):
                for k in range(eri.shape[2]):
                    for l in range(eri.shape[3]):
                        if abs(eri_bench[i,j,k,l] - eri[i,j,k,l]) > tol:
                            print("eri[{}, {}, {}, {}] = {} != {}".format(i,j,k,l,eri_bench[i,j,k,l], eri[i,j,k,l]),
                                  "ration = ", eri_bench[i,j,k,l]/eri[i,j,k,l])

from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition

if __name__ == '__main__':

    # for c in [3, 5, 10, 15]:
    for c in [25]:
        # for N in [1, 2]:
        # for N in [2]:
        for N in [1]:

            print("Testing c = ", c, "N = ", N, "...")

            cell   = pbcgto.Cell()
            boxlen = 3.5668
            cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
            
            # cell.atom = '''
            #                C     0.8917  0.8917  0.8917
            #                C     2.6751  2.6751  0.8917
            #                C     2.6751  0.8917  2.6751
            #                C     0.8917  2.6751  2.6751
            #             '''

            # cell.atom = '''
            #                C     0.      0.      0.
            #                C     0.8917  0.8917  0.8917
            #                C     1.7834  1.7834  0.
            #                C     2.6751  2.6751  0.8917
            #                C     1.7834  0.      1.7834
            #                C     2.6751  0.8917  2.6751
            #                C     0.      1.7834  1.7834
            #                C     0.8917  2.6751  2.6751
            #             '''
            
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
            # cell.build()
            
            verbose = 4
            
            prim_cell = build_supercell(cell.atom, cell.a, Ls = [1,1,1], ke_cutoff=cell.ke_cutoff, basis=cell.basis, pseudo=cell.pseudo)   
            # prim_partition = [[0,1,2,3,4,5,6,7]]
            # prim_partition = [[0,1],[2,3],[4,5],[6,7]]
            prim_partition = [[0,1,2,3], [4,5,6,7]]
            prim_mesh = prim_cell.mesh
            
            Ls = [1, 1, N]
            Ls = np.array(Ls, dtype=np.int32)
            mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
            mesh = np.array(mesh, dtype=np.int32)
            
            # cell = tools.super_cell(cell, [1, 1, N])
            
            cell, group_partition = build_supercell_with_partition(
                                    cell.atom, cell.a, mesh=mesh, 
                                    Ls=Ls,
                                    basis=cell.basis, 
                                    pseudo=cell.pseudo,
                                    partition=prim_partition, ke_cutoff=cell.ke_cutoff, verbose=verbose)

            # myisdf = ISDF(cell=cell)
            # myisdf.build(c=c)
            
            myisdf = isdf_linear_scaling.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
            myisdf.build_IP_local(c=c, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
            # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*3, Ls[1]*3, Ls[2]*3])
            # pbc_isdf_info.Ls = Ls
            myisdf.build_auxiliary_Coulomb(debug=True)

            from pyscf.pbc import scf
            mf = scf.RHF(cell)
            myisdf.direct_scf = mf.direct_scf
            mf.with_df = myisdf
            mf.max_cycle = 8
            mf.conv_tol = 1e-7
            mf.kernel()

            ######## ao eri benchmark ########

            mydf_eri = df.FFTDF(cell)

            eri = mydf_eri.get_eri(compact=True)
            eri_isdf = myisdf.get_eri(compact=True)
            _check_eri(eri, eri_isdf)
            
            eri = mydf_eri.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            eri_isdf = myisdf.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            _check_eri(eri, eri_isdf)
                        
            # continue

            #### mo eri benchmark ########

            # mo_coeff = np.random.random((cell.nao,cell.nao))

            mo_coeff = mf.mo_coeff
            nocc     = cell.nelectron // 2
            
            mo_coeff_o = mo_coeff[:, :nocc].copy()
            mo_coeff_v = mo_coeff[:, nocc:].copy()

            ######## compact = False ########

            mo_eri = mydf_eri.ao2mo(mo_coeff, compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            mo_eri_isdf = myisdf.ao2mo(mo_coeff, compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            _check_eri(mo_eri, mo_eri_isdf)
            
            mo_eri_4_fold = mo_eri.copy()

            ######## compact = True ########

            mo_eri = mydf_eri.ao2mo(mo_coeff, compact=True)
            mo_eri_isdf = myisdf.ao2mo(mo_coeff, compact=True)
            _check_eri(mo_eri, mo_eri_isdf)

            ######## test ovov ######## 
            
            mo_eri_ovov = mydf_eri.ao2mo((mo_coeff_o,mo_coeff_v,mo_coeff_o,mo_coeff_v), compact=False).reshape(nocc, cell.nao-nocc, nocc, cell.nao-nocc) # why it is not correct ? 
            
            mo_eri_benchmark = mo_eri_4_fold[:nocc, nocc:, :nocc, nocc:].copy()
            diff = np.linalg.norm(mo_eri_ovov - mo_eri_benchmark)
            print("diff = ", diff)
            # assert np.allclose(mo_eri_ovov, mo_eri_benchmark)
            
            mo_eri_ovov_isdf = myisdf.ao2mo((mo_coeff_o,mo_coeff_v,mo_coeff_o,mo_coeff_v), compact=False).reshape(nocc, cell.nao-nocc, nocc, cell.nao-nocc)
            _check_eri(mo_eri_ovov_isdf, mo_eri_benchmark)