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

from pyscf import gto, scf

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.ao2mo import eris
import pyscf.pbc.tools

import pyscf.pbc.cc

import pyscf.pbc.tools.make_test_cell as make_test_cell

def run_cell(cell, n, nk):
    #############################################
    # Do a supercell Gamma-pt calculation       #
    #############################################
    supcell = pyscf.pbc.tools.super_cell(cell, nk)
    supcell.build()
    gamma = [0,0,0]

    mf = pbchf.RHF(supcell, exxdiv=None)
    mf.conv_tol = 1e-14
    #mf.verbose = 7
    escf = mf.scf()
    escf_per_cell = escf/np.prod(nk)

    cc = pyscf.pbc.cc.CCSD(mf)
    cc.conv_tol=1e-8
    ecc, t1, it2 = cc.kernel()
    ecc_per_cell = ecc/np.prod(nk)
    return escf_per_cell, ecc_per_cell

class KnownValues(unittest.TestCase):
    def test_111_n0(self):
        L = 10.0
        n = 11
        cell = make_test_cell.test_cell_n0(L,[n]*3)
        nk = (1, 1, 1)
        hf_111 = -0.73491491306419987
        cc_111 = -1.1580008204825658e-05
        escf,ecc=run_cell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111, 7)
        self.assertAlmostEqual(ecc, cc_111, 6)

    def test_111_n1(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (1, 1, 1)
        hf_111 = -0.73506011036963814
        cc_111 = -0.023265431169472835
        escf,ecc=run_cell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111, 7)
        self.assertAlmostEqual(ecc, cc_111, 6)

    def test_111_n3(self):
        L = 10.0
        n = 11
        cell = make_test_cell.test_cell_n3([n]*3)
        nk = (1, 1, 1)
        hf_111 = -7.4117951240232118
        cc_111 = -0.19468901057053406
        escf,ecc=run_cell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111, 7)
        self.assertAlmostEqual(ecc, cc_111, 6)

    def test_311_n1(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf,ecc=run_cell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_311, 7)
        self.assertAlmostEqual(ecc, cc_311, 6)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()
