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

    def test_uagf2_slow_beh_1_0(self):
        # tests AGF2(None,0) for BeH/cc-pvdz
        mol = gto.M(atom='Be 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.run()

        gf2 = agf2.uagf2_slow.UAGF2(mf, nmom=(None,0))
        gf2.conv_tol = 1e-7
        gf2.run()
        self.assertTrue(gf2.converged)
        self.assertAlmostEqual(mf.e_tot,  -15.0910903300424    , 10)
        self.assertAlmostEqual(gf2.e_1b,  -15.069681001221705  ,  6)
        self.assertAlmostEqual(gf2.e_2b,  -0.049461593728309786,  6)
        self.assertAlmostEqual(gf2.e_init, -0.025198374705580943,  6)

        e_ip, v_ip = gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.3003522286132736, 6)
        self.assertAlmostEqual(e_ip[1], 0.5107596660196604, 6)
        self.assertAlmostEqual(e_ip[2], 0.5318094633979558, 6)
        self.assertAlmostEqual(v_ip[0], 0.9962231685493768, 6)
        self.assertAlmostEqual(v_ip[1], 0.9789822411853315, 6)
        self.assertAlmostEqual(v_ip[2], 0.9809062972345126, 6)

        e_ea, v_ea = gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.03781071654337435, 6)
        self.assertAlmostEqual(e_ea[1], 0.04252189700736402, 6)
        self.assertAlmostEqual(e_ea[2], 0.0425218970073656 , 6)
        self.assertAlmostEqual(v_ea[0], 0.9740024912068087 , 6)
        self.assertAlmostEqual(v_ea[1], 0.9902310149008003 , 6)
        self.assertAlmostEqual(v_ea[2], 0.9902310149008006 , 6)

    def test_uagf2_slow_beh_2_3(self):
        # tests AGF2(2,3) for BeH/6-31g
        mol = gto.M(atom='Be 0 0 0; H 0 0 1', basis='6-31g', spin=1, verbose=0)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.run()

        gf2 = agf2.uagf2_slow.UAGF2(mf, nmom=(2,3))
        gf2.conv_tol = 1e-7
        gf2.run()
        self.assertTrue(gf2.converged)
        self.assertAlmostEqual(gf2.e_init, -0.0153603737842962, 4)


if __name__ == '__main__':
    print('UAGF2 calculations for BeH')
    unittest.main()
