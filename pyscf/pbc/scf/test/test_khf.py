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
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import kuhf
import pyscf.pbc.tools

def finger(a):
    return np.dot(np.cos(np.arange(a.size)), a.ravel())

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

    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def test_kpt_vs_supercell_high_cost(self):
        # For large n, agreement is always achieved
        # n = 17
        # For small n, agreement only achieved if "wrapping" k-k'+G in get_coulG
        n = 9
        nk = (3, 1, 1)
        cell = make_primitive_cell([n]*3)

        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        self.assertAlmostEqual(ekpt, -11.221426249047617, 8)

        nk = (5, 1, 1)
        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        self.assertAlmostEqual(ekpt, -12.337299166550796, 8)

        supcell = pyscf.pbc.tools.super_cell(cell, nk)
        supcell.build()
        #print "supcell mesh =", supcell.mesh

        gamma = [0,0,0]
        mf = khf.KRHF(supcell, gamma, exxdiv='vcut_sph')
        esup = mf.scf()/np.prod(nk)

        #print "kpt sampling energy =", ekpt
        #print "supercell energy    =", esup
        #print "difference          =", ekpt-esup
        self.assertAlmostEqual(ekpt, esup, 6)

    def test_init_guess_by_chkfile(self):
        n = 9
        nk = (1, 1, 1)
        cell = make_primitive_cell([n]*3)

        kpts = cell.make_kpts(nk)
        kmf = khf.KRHF(cell, kpts, exxdiv='vcut_sph')
        kmf.conv_tol = 1e-9
        ekpt = kmf.scf()
        dm1 = kmf.make_rdm1()
        dm2 = kmf.from_chk(kmf.chkfile)
        self.assertTrue(dm2.dtype == np.double)
        self.assertTrue(np.allclose(dm1, dm2))

        mf = pbchf.RHF(cell, exxdiv='vcut_sph')
        mf.chkfile = kmf.chkfile
        mf.init_guess = 'chkfile'
        dm1 = mf.from_chk(kmf.chkfile)
        mf.max_cycle = 1
        e1 = mf.kernel(dm1)
        mf.conv_check = False
        self.assertAlmostEqual(e1, ekpt, 9)

        nk = (3, 1, 1)
        kpts = cell.make_kpts(nk)
        kmf1 = khf.KRHF(cell, kpts, exxdiv='vcut_sph')
        kmf1.conv_tol = 1e-9
        dm = kmf1.from_chk(mf.chkfile)
        kmf1.max_cycle = 2
        ekpt = kmf1.scf(dm)
        kmf1.conv_check = False
        self.assertAlmostEqual(ekpt, -11.215218432275057, 8)

    def test_kuhf_high_cost(self):
        n = 9
        cell = make_primitive_cell([n]*3)
        nk = (3, 1, 1)
        kpts = cell.make_kpts(nk)
        kmf1 = kuhf.KUHF(cell, kpts, exxdiv='vcut_sph')
        kmf1.conv_tol = 1e-9
        ekpt = kmf1.scf()
        self.assertAlmostEqual(ekpt, -11.218735269838586, 8)
        np.random.seed(1)
        kpts_bands = np.random.random((2,3))
        e = kmf1.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(finger(np.array(e)), -0.045547555445877741, 6)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf")
    unittest.main()
