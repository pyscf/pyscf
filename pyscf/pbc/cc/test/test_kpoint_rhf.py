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
    def test_311_n1(self):
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
        mf = pbcscf.KRHF(cell, kpts=kpts).run()
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
        ehf = kmf.scf()

        # KRCCSD calculation, equivalent to running supercell
        # calculation with frozen=[0,1,2] (if done with larger mesh)
        cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf, frozen=[[0],[0,1]])
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ehf, ehf_bench, 9)
        self.assertAlmostEqual(ecc, ecc_bench, 9)

if __name__ == '__main__':
    print("Full kpoint_rhf test")
    unittest.main()

