# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
        self.assertAlmostEqual(gf2.e_1b,  -15.0696810889066    ,  6)
        self.assertAlmostEqual(gf2.e_2b,  -0.04953132465388112 ,  6)
        self.assertAlmostEqual(gf2.e_mp2, -0.025198374705580943,  6)

        e_ip, v_ip = gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.2997045882193411, 6)
        self.assertAlmostEqual(e_ip[1], 0.5061695884140355, 6)
        self.assertAlmostEqual(e_ip[2], 0.5284717393681077, 6)
        self.assertAlmostEqual(v_ip[0], 0.9960125215054924, 6)
        self.assertAlmostEqual(v_ip[1], 0.9758519647601754, 6)
        self.assertAlmostEqual(v_ip[2], 0.9790675232737558, 6)

        e_ea, v_ea = gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.04238166572514309, 6)
        self.assertAlmostEqual(e_ea[1], 0.04402081113588716, 6)
        self.assertAlmostEqual(e_ea[2], 0.04402081113588819, 6)
        self.assertAlmostEqual(v_ea[0], 0.9808256857147465 , 6)
        self.assertAlmostEqual(v_ea[1], 0.9917595650497684 , 6)
        self.assertAlmostEqual(v_ea[2], 0.9917595650497666 , 6)

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
        self.assertAlmostEqual(gf2.e_mp2, -0.0153603737842962, 4)
        #TODO


if __name__ == '__main__':
    print('UAGF2 calculations for BeH')
    unittest.main()
