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


import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
import pyscf.pbc.mp
import pyscf.pbc.mp.kmp2
from pyscf.df import make_auxbasis


def build_cell(space_group_symmetry=False):
    # open-shell system may lead to larger uncertainty than required precision
    # atom = 'C 0 0 0'
    atom = 'Be 0 0 0'
    a = np.eye(3) * 5
    basis = 'cc-pvdz'
    if space_group_symmetry:
        return pbcgto.M(atom=atom, basis=basis, a=a, precision=1e-8, verbose=4,
                        output='/dev/null',
                        space_group_symmetry=True, symmorphic=False)
    return pbcgto.M(atom=atom, basis=basis, a=a, precision=1e-8, verbose=4,
                    output='/dev/null')


class KnownValues(unittest.TestCase):
    def test_mp2(self):
        cell = build_cell()
        mf = pbcscf.RHF(cell).density_fit(auxbasis=make_auxbasis(cell))
        mf.conv_tol = 1e-10
        mf.kernel()
        pt = pyscf.pbc.mp.mp2.RMP2(mf).run()

        self.assertAlmostEqual(pt.e_corr, -0.016306542341971778, 7)
        self.assertAlmostEqual(pt.e_corr_ss, -4.3676116478926635e-05, 7)
        self.assertAlmostEqual(pt.e_corr_os, -0.01626286622549285, 7)

    def test_kmp2(self):
        def run_k(cell, kmesh):
            kpts = cell.make_kpts(kmesh)
            mf = pbcscf.KRHF(cell, kpts).density_fit(auxbasis=make_auxbasis(cell))
            mf.conv_tol = 1e-10
            mf.kernel()
            pt = pyscf.pbc.mp.kmp2.KMP2(mf).run()
            return pt

        cell = build_cell()

        pt = run_k(cell, (1,1,1))
        self.assertAlmostEqual(pt.e_corr, -0.016306542341971778, 7)
        self.assertAlmostEqual(pt.e_corr_ss, -4.3676116478926635e-05, 7)
        self.assertAlmostEqual(pt.e_corr_os, -0.01626286622549285, 7)

        pt = run_k(cell, (2,1,1))
        self.assertAlmostEqual(pt.e_corr, -0.022116013287498435, 7)
        self.assertAlmostEqual(pt.e_corr_ss, -0.0006375312743462651, 7)
        self.assertAlmostEqual(pt.e_corr_os, -0.02147848201315217, 7)

    def test_ksymm(self):
        def run_k(cell, kmesh):
            kpts = cell.make_kpts(kmesh, space_group_symmetry=True)
            mf = pbcscf.KRHF(cell, kpts).density_fit(auxbasis=make_auxbasis(cell))
            mf.conv_tol = 1e-10
            mf.kernel()
            pt = pyscf.pbc.mp.kmp2_ksymm.KMP2(mf).run()
            return pt

        cell = build_cell(space_group_symmetry=True)

        pt = run_k(cell, (1,1,1))
        self.assertAlmostEqual(pt.e_corr, -0.016306542341971778, 7)
        self.assertAlmostEqual(pt.e_corr_ss, -4.3676116478926635e-05, 7)
        self.assertAlmostEqual(pt.e_corr_os, -0.01626286622549285, 7)

        pt = run_k(cell, (2,1,1))
        self.assertAlmostEqual(pt.e_corr, -0.022116013287498435, 7)
        self.assertAlmostEqual(pt.e_corr_ss, -0.0006375312743462651, 7)
        self.assertAlmostEqual(pt.e_corr_os, -0.02147848201315217, 7)


if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()
