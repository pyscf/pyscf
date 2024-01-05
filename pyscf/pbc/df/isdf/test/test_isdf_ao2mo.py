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

from pyscf.pbc.df.isdf import ISDF

import numpy as np
from pyscf import ao2mo
from pyscf.pbc.df import aft
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc import tools
from pyscf.pbc import df

def _check_eri(eri, eri_bench, tol=1e-6):
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

if __name__ == '__main__':

    for c in [3, 5, 10, 15]:
        for N in [1, 2]:

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

            cell.basis   = 'gth-szv'
            cell.pseudo  = 'gth-pade'
            cell.verbose = 4

            # cell.ke_cutoff  = 100   # kinetic energy cutoff in a.u.
            cell.ke_cutoff = 128
            cell.max_memory = 800  # 800 Mb
            cell.precision  = 1e-8  # integral precision
            cell.use_particle_mesh_ewald = True

            cell.build()

            cell = tools.super_cell(cell, [1, 1, 1])

            myisdf = ISDF(cell=cell)
            myisdf.build(c=10)

            ######## ao eri benchmark ########

            mydf_eri = df.FFTDF(cell)

            eri = mydf_eri.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            eri_isdf = myisdf.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            _check_eri(eri, eri_isdf)

            eri = mydf_eri.get_eri(compact=True)
            eri_isdf = myisdf.get_eri(compact=True)
            _check_eri(eri, eri_isdf)

            #### mo eri benchmark ########

            mo_coeff = np.random.random((cell.nao,cell.nao))

            ######## compact = False ########

            mo_eri = mydf_eri.ao2mo(mo_coeff, compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            mo_eri_isdf = myisdf.ao2mo(mo_coeff, compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
            _check_eri(mo_eri, mo_eri_isdf)

            ######## compact = True ########

            mo_eri = mydf_eri.ao2mo(mo_coeff, compact=True)
            mo_eri_isdf = myisdf.ao2mo(mo_coeff, compact=True)
            _check_eri(mo_eri, mo_eri_isdf)
