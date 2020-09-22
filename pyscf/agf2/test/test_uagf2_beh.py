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
        self.mol = gto.M(atom='Be 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        self.mf = scf.UHF(self.mol)
        self.mf.conv_tol = 1e-12
        self.mf.run()
        self.gf2 = agf2.UAGF2(self.mf)
        self.gf2.conv_tol = 1e-7
        self.gf2.run()

    @classmethod
    def tearDownClass(self):
        del self.mol, self.mf, self.gf2
        
    def test_uagf2_beh_ground_state(self):
        # tests the ground state AGF2 energies for BeH/cc-pvdz
        self.assertTrue(self.gf2.converged)
        self.assertAlmostEqual(self.mf.e_tot,  -15.0910903300424    , 10)
        self.assertAlmostEqual(self.gf2.e_1b,  -15.0696810889066    ,  6)
        self.assertAlmostEqual(self.gf2.e_2b,  -0.04953132465388112 ,  6)
        self.assertAlmostEqual(self.gf2.e_mp2, -0.025198374705580943,  6)

    def test_uagf2_beh_ip(self):
        # tests the AGF2 ionization potentials for BeH/cc-pvdz
        e_ip, v_ip = self.gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.2997045882193411, 6)
        self.assertAlmostEqual(e_ip[1], 0.5061695884140355, 6)
        self.assertAlmostEqual(e_ip[2], 0.5284717393681077, 6)
        self.assertAlmostEqual(v_ip[0], 0.9960125215054924, 6)
        self.assertAlmostEqual(v_ip[1], 0.9758519647601754, 6)
        self.assertAlmostEqual(v_ip[2], 0.9790675232737558, 6)

    def test_uagf2_beh_ea(self):
        # tests the AGF2 electron affinities for BeH/cc-pvdz
        e_ea, v_ea = self.gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.04238166572514309, 6)
        self.assertAlmostEqual(e_ea[1], 0.04402081113588716, 6)
        self.assertAlmostEqual(e_ea[2], 0.04402081113588819, 6)
        self.assertAlmostEqual(v_ea[0], 0.9808256857147465 , 6)
        self.assertAlmostEqual(v_ea[1], 0.9917595650497684 , 6)
        self.assertAlmostEqual(v_ea[2], 0.9917595650497666 , 6)

    def test_uagf2_outcore(self):
        # tests the out-of-core and chkfile support for AGF2 for BeH/cc-pvdz
        gf2 = agf2.UAGF2(self.mf)
        gf2.max_memory = 1
        gf2.conv_tol = 1e-7
        gf2.run()
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(self.gf2.e_1b, -15.0696810889066   , 6)
        self.assertAlmostEqual(self.gf2.e_2b, -0.04953132465388112, 6)
        self.assertAlmostEqual(e_ip,          0.2997045882193411  , 6)
        self.assertAlmostEqual(v_ip,          0.9960125215054924  , 6)
        self.assertAlmostEqual(e_ea,          0.04238166572514309 , 6)
        self.assertAlmostEqual(v_ea,          0.9808256857147465  , 6)
        gf2.dump_chk()
        gf2 = agf2.UAGF2(self.mf)
        gf2.__dict__.update(lib.chkfile.load(gf2.chkfile, 'agf2'))
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(self.gf2.e_1b, -15.0696810889066   , 6)
        self.assertAlmostEqual(self.gf2.e_2b, -0.04953132465388112, 6)
        self.assertAlmostEqual(e_ip,          0.2997045882193411  , 6)
        self.assertAlmostEqual(v_ip,          0.9960125215054924  , 6)
        self.assertAlmostEqual(e_ea,          0.04238166572514309 , 6)
        self.assertAlmostEqual(v_ea,          0.9808256857147465  , 6)


if __name__ == '__main__':
    print('UAGF2 calculations for BeH')
    unittest.main()
