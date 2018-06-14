#!/usr/bin/env python
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
#
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf

import pyscf.cc
import pyscf.pbc.cc
import pyscf.pbc.cc.kccsd_rhf
import pyscf.pbc.cc.ccsd
import make_test_cell

import pyscf.pbc.cc.kccsd_t_rhf as kccsd_t_rhf

def run_kcell(cell, n, nk):
    #############################################
    # Do a k-point calculation                  #
    #############################################
    abs_kpts = cell.make_kpts(nk, wrap_around=True)

    #############################################
    # Running HF                                #
    #############################################
    kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-14
    #kmf.verbose = 7
    ekpt = kmf.scf()


    cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf)
    cc.conv_tol=1e-8
    cc.verbose = 7
    ecc, t1, t2 = cc.kernel()
    return ekpt, ecc

class KnownValues(unittest.TestCase):
    def test_311_n1_high_cost(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

    def test_single_kpt(self):
        cell = pbcgto.Cell()
        cell.atom = '''
        H 0 0 0
        H 1 0 0
        H 0 1 0
        H 0 1 1
        '''
        cell.a = np.eye(3)*2
        cell.basis = [[0, [1.2, 1]], [1, [1.0, 1]]]
        cell.verbose = 0
        cell.build()

        kpts = cell.get_abs_kpts([.5,.5,.5]).reshape(1,3)
        mf = pbcscf.KRHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kcc = pyscf.pbc.cc.kccsd_rhf.RCCSD(mf)
        e0 = kcc.kernel()[0]

        mf = pbcscf.RHF(cell, kpt=kpts[0]).run()
        mycc = pyscf.pbc.cc.RCCSD(mf)
        e1 = mycc.kernel()[0]
        self.assertAlmostEqual(e0, e1, 7)

    def test_frozen_n3(self):
        mesh = 5
        cell = make_test_cell.test_cell_n3([mesh]*3)
        nk = (1, 1, 2)
        ehf_bench = -8.348616843863795
        ecc_bench = -0.037920339437169

        abs_kpts = cell.make_kpts(nk, with_gamma_point=True)

        # RHF calculation
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-9
        ehf = kmf.scf()

        # KRCCSD calculation, equivalent to running supercell
        # calculation with frozen=[0,1,2] (if done with larger mesh)
        cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=[[0],[0,1]])
        cc.diis_start_cycle = 1
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ehf, ehf_bench, 9)
        self.assertAlmostEqual(ecc, ecc_bench, 9)

    def _test_cu_metallic_nonequal_occ(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc1_bench = -0.9646107739333411
        max_cycle = 5  # Too expensive to do more

        # The following calculation at full convergence gives -0.711071910294612
        # for a cell.mesh = [25, 25, 25].
        mycc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=0)
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.05
        mycc.max_cycle = max_cycle
        ecc1, t1, t2 = mycc.kernel()

        self.assertAlmostEqual(ecc1, ecc1_bench, 6)

    def _test_cu_metallic_frozen_occ(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc2_bench = -0.7651806468801496
        max_cycle = 5

        # The following calculation at full convergence gives -0.6440448716452378
        # for a cell.mesh = [25, 25, 25].  It is equivalent to an RHF supercell [1, 1, 2]
        # calculation with frozen = [0, 3].
        mycc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=[[2, 3], [0, 1]])
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.05
        mycc.max_cycle = max_cycle
        ecc2, t1, t2 = mycc.kernel()

        self.assertAlmostEqual(ecc2, ecc2_bench, 6)

    def _test_cu_metallic_frozen_vir(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc3_bench = -0.76794053711557086
        max_cycle = 5

        # The following calculation at full convergence gives -0.58688462599474
        # for a cell.mesh = [25, 25, 25].  It is equivalent to a supercell [1, 1, 2]
        # calculation with frozen = [0, 3, 35].
        mycc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=[[1, 17], [0]])
        mycc.diis_start_cycle = 1
        mycc.max_cycle = max_cycle
        mycc.iterative_damping = 0.05
        ecc3, t1, t2 = mycc.kernel()

        self.assertAlmostEqual(ecc3, ecc3_bench, 6)

        check_gamma = False  # Turn me on to run the supercell calculation!

        if check_gamma:
            from pyscf.pbc.tools.pbc import super_cell
            supcell = super_cell(cell, nk)
            kmf = pbcscf.RHF(supcell, exxdiv=None)
            ehf = kmf.scf()

            mycc = pyscf.pbc.cc.RCCSD(kmf, frozen=[0, 3, 35])
            mycc.max_cycle = max_cycle
            mycc.iterative_damping = 0.05
            ecc, t1, t2 = mycc.kernel()

            print('Gamma energy =', ecc/np.prod(nk))
            print('K-point energy =', ecc3)

    def test_cu_metallic_high_cost(self):
        mesh = 7
        cell = make_test_cell.test_cell_cu_metallic([mesh]*3)
        nk = [1,1,2]
        ehf_bench = -52.5393701339723

        # KRHF calculation
        kmf = pbcscf.KRHF(cell, exxdiv=None)
        kmf.kpts = cell.make_kpts(nk, scaled_center=[0.0, 0.0, 0.0], wrap_around=True)
        kmf.conv_tol_grad = 1e-6  # Stricter tol needed for answer to agree with supercell
        ehf = kmf.scf()

        self.assertAlmostEqual(ehf, ehf_bench, 6)

        # Run CC calculations
        self._test_cu_metallic_nonequal_occ(kmf, cell, nk=nk)
        self._test_cu_metallic_frozen_occ(kmf, cell, nk=nk)
        self._test_cu_metallic_frozen_vir(kmf, cell, nk=nk)

    def test_ccsd_t_high_cost(self):
        n = 14
        cell = make_test_cell.test_cell_n3([n]*3)

        kpts = cell.make_kpts([1, 1, 2])
        kpts -= kpts[0]
        kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None)
        ehf = kmf.kernel()

        mycc = pyscf.pbc.cc.KRCCSD(kmf)
        ecc, t1, t2 = mycc.kernel()

        energy_t = kccsd_t_rhf.kernel(mycc)
        energy_t_bench = -0.00191443154358
        self.assertAlmostEqual(energy_t, energy_t_bench, 6)

if __name__ == '__main__':
    print("Full kpoint_rhf test")
    unittest.main()

