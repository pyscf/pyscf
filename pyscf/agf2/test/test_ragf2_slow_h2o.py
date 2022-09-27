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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

import unittest
import numpy as np
from pyscf import gto, scf, agf2, lib


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_ragf2_slow_h2o_1_0(self):
        # tests AGF2(None,0) for H2O/cc-pvdz
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.run()

        gf2 = agf2.ragf2_slow.RAGF2(mf, nmom=(None,0))
        gf2.conv_tol = 1e-7
        gf2.run()
        self.assertTrue(gf2.converged)
        self.assertAlmostEqual(mf.e_tot,  -76.0167894720742   , 10)
        self.assertAlmostEqual(gf2.e_1b,  -75.89108074396137  ,  6)
        self.assertAlmostEqual(gf2.e_2b,  -0.33248785652834784,  6)
        self.assertAlmostEqual(gf2.e_init, -0.17330473289845347,  6)

        e_ip, v_ip = gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.45080222600137465, 6)
        self.assertAlmostEqual(e_ip[1], 0.5543195106668687 , 6)
        self.assertAlmostEqual(e_ip[2], 0.6299640547362962 , 6)
        self.assertAlmostEqual(v_ip[0], 0.9704061235804103 , 6)
        self.assertAlmostEqual(v_ip[1], 0.9702372037466642 , 6)
        self.assertAlmostEqual(v_ip[2], 0.9713854565834782 , 6)

        e_ea, v_ea = gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.15581330758457984, 6)
        self.assertAlmostEqual(e_ea[1], 0.2347918376963518 , 6)
        self.assertAlmostEqual(e_ea[2], 0.686105303143818  , 6)
        self.assertAlmostEqual(v_ea[0], 0.9903734898112396 , 6)
        self.assertAlmostEqual(v_ea[1], 0.9901410412716749 , 6)
        self.assertAlmostEqual(v_ea[2], 0.9827713231118138 , 6)

    def test_ragf2_slow_lih_3_4(self):
        # tests AGF2(3,4) for LiH/cc-pvdz
        mol = gto.M(atom='Li 0 0 0; H 0 0 1', basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.run()

        gf2 = agf2.ragf2_slow.RAGF2(mf, nmom=(3,4))
        gf2.conv_tol = 1e-7
        gf2.run()
        self.assertTrue(gf2.converged)
        self.assertAlmostEqual(gf2.e_init, -0.029669047726821392, 4)

    def test_moments(self):
        # tests conservation of moments with compression
        mol = gto.M(atom='Li 0 0 0; H 0 0 1', basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.run()

        gf2 = agf2.RAGF2(mf, nmom=(3,4)).run(max_cycle=3)
        eri = gf2.ao2mo()
        gf = gf2.gf
        gf_occ, gf_vir = gf.get_occupied(), gf.get_virtual()
        se_occ = agf2.ragf2_slow.build_se_part(gf2, eri, gf_occ, gf_vir)
        se_vir = agf2.ragf2_slow.build_se_part(gf2, eri, gf_vir, gf_occ)
        se = agf2.aux.combine(se_occ, se_vir)

        def count_correct_moms(a, b, tol=1e-9):
            n = 0
            while True:
                ma = a.moment(n)
                mb = b.moment(n)
                dif = (ma - mb) / np.mean(np.absolute(0.5 * (ma + mb)))
                err = np.linalg.norm(dif)
                if err < tol:
                    n += 1
                else:
                    break
            return n

        se_s3 = se.compress(phys=None, n=(None,3))
        self.assertEqual(count_correct_moms(se, se_s3), 8)

        se_s4 = se.compress(phys=None, n=(None,4))
        self.assertEqual(count_correct_moms(se, se_s4), 10)

        gf = se_s4.get_greens_function(eri.fock)
        se_g3 = se_s4.compress(phys=eri.fock, n=(3,None))
        gf_g3 = se_g3.get_greens_function(eri.fock)
        self.assertEqual(count_correct_moms(gf, gf_g3), 8)

        gf = se_s4.get_greens_function(eri.fock)
        se_g4 = se_s4.compress(phys=eri.fock, n=(4,None))
        gf_g4 = se_g4.get_greens_function(eri.fock)
        self.assertEqual(count_correct_moms(gf, gf_g4), 10)


if __name__ == '__main__':
    print('RAGF2 calculations for H2O')
    unittest.main()
