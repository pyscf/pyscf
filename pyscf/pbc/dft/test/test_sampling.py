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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import unittest
import numpy as np

import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.tools

def make_primitive_cell(mesh):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh

    #cell.nimgs = np.array([7,7,7])
    cell.verbose = 0
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def xtest_gamma(self):
        cell = make_primitive_cell([17]*3)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -10.2214263103747, 8)

    def xtest_kpt_222(self):
        cell = make_primitive_cell([17]*3)
        abs_kpts = cell.make_kpts([2,2,2], wrap_around=True)
        kmf = pbcdft.KRKS(cell, abs_kpts)
        kmf.xc = 'lda,vwn'
        #kmf.analytic_int = False
        #kmf.verbose = 7
        e1 = kmf.scf()
        self.assertAlmostEqual(e1, -11.3536435234900, 8)

    def test_kpt_vs_supercell(self):
        n = 11
        nk = (3, 1, 1)
        # Comparison is only perfect for odd-numbered supercells and kpt sampling
        assert all(np.array(nk) % 2 == np.array([1,1,1]))
        cell = make_primitive_cell([n]*3)
        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pbcdft.KRKS(cell, abs_kpts)
        kmf.xc = 'lda,vwn'
        #kmf.analytic_int = False
        #kmf.verbose = 7
        ekpt = kmf.scf()

        supcell = pyscf.pbc.tools.super_cell(cell, nk)
        supcell.build()

        mf = pbcdft.RKS(supcell)
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        esup = mf.scf()/np.prod(nk)

        #print("kpt sampling energy =", ekpt)
        #print("supercell energy    =", esup)
        self.assertAlmostEqual(ekpt, esup, 5)


if __name__ == '__main__':
    print("Full Tests for k-point sampling")
    unittest.main()
