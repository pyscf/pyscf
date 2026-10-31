#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

from pyscf.pbc import gto, scf, mp


def setUpModule():
    global cell
    cell = gto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'gth-szv'
    cell.ke_cutoff = 100
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.build()

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell


class KnownValues(unittest.TestCase):
    def _check_blksize(self, kmf, with_df_ints, e_ref, e_ss_ref, e_os_ref):
        for blksize in (1, 2, 3):
            with self.subTest(blksize=blksize):
                mymp = mp.KMP2(kmf)
                mymp.blksize = blksize
                self.assertEqual(mymp.with_df_ints, with_df_ints)
                e, _ = mymp.kernel(with_t2=False)
                self.assertEqual(mymp.nocc, 4)
                occ_blocks = [(i0, min(i0+blksize, 4)) for i0 in range(0, 4, blksize)]
                self.assertEqual(mymp._occ_blocks, occ_blocks)
                self.assertAlmostEqual(e, e_ref, 7)
                self.assertAlmostEqual(mymp.e_corr_ss, e_ss_ref, 7)
                self.assertAlmostEqual(mymp.e_corr_os, e_os_ref, 7)

    def test_kmp2_gdf(self):
        def run_scf(kmesh):
            kpts = cell.make_kpts(kmesh)
            kmf = scf.KRHF(cell, kpts).density_fit()
            kmf.conv_tol = 1e-10
            kmf.kernel()
            return kmf

        mymf = run_scf([1, 1, 1])
        self._check_blksize(mymf, True, -0.1090404700798, -0.0158026100138, -0.0932378600660)

        mymf = run_scf([2, 1, 1])
        self._check_blksize(mymf, True, -0.1085115957829, -0.0170425395647, -0.0914690562182)

    def test_kmp2_fftdf(self):
        def run_scf(kmesh):
            kpts = cell.make_kpts(kmesh)
            kmf = scf.KRHF(cell, kpts)
            kmf.conv_tol = 1e-10
            kmf.kernel()
            return kmf

        mymf = run_scf([1, 1, 1])
        self._check_blksize(mymf, False, -0.1115097882396, -0.0172099643724, -0.0942998238671)

        mymf = run_scf([2, 1, 1])
        self._check_blksize(mymf, False, -0.1098784304579, -0.0177706849278, -0.0921077455301)


if __name__ == '__main__':
    print("KMP2 batching tests")
    unittest.main()
