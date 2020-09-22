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
        self.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        self.mf = scf.RHF(self.mol)
        self.mf.conv_tol = 1e-12
        self.mf.run()
        self.gf2 = agf2.RAGF2(self.mf)
        self.gf2.conv_tol = 1e-7
        self.gf2.run()

    @classmethod
    def tearDownClass(self):
        del self.mol, self.mf, self.gf2

    def test_ragf2_h2o_ground_state(self):
        # tests the ground state AGF2 energies for H2O/cc-pvdz
        self.assertTrue(self.gf2.converged)
        self.assertAlmostEqual(self.mf.e_tot,  -76.0167894720742   , 10)
        self.assertAlmostEqual(self.gf2.e_1b,  -75.89108074653284  ,  6)
        self.assertAlmostEqual(self.gf2.e_2b,  -0.3323976129158455 ,  6)
        self.assertAlmostEqual(self.gf2.e_mp2, -0.17330473289845347,  6)

    def test_ragf2_h2o_ip(self):
        # tests the AGF2 ionization potentials for H2O/cc-pvdz
        e_ip, v_ip = self.gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.4463033078563678, 6)
        self.assertAlmostEqual(e_ip[1], 0.5497970633171162, 6)
        self.assertAlmostEqual(e_ip[2], 0.6256188543981303, 6)
        self.assertAlmostEqual(v_ip[0], 0.9690389634580812, 6)
        self.assertAlmostEqual(v_ip[1], 0.9688865172211362, 6)
        self.assertAlmostEqual(v_ip[2], 0.9701094049601187, 6)

    def test_ragf2_h2o_ea(self):
        # tests the AGF2 electron affinities for H2O/cc-pvdz
        e_ea, v_ea = self.gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.1572194156451459, 6)
        self.assertAlmostEqual(e_ea[1], 0.236241405622162 , 6)
        self.assertAlmostEqual(e_ea[2], 0.6886009727626674, 6)
        self.assertAlmostEqual(v_ea[0], 0.9906842311023561, 6)
        self.assertAlmostEqual(v_ea[1], 0.9903372215743906, 6)
        self.assertAlmostEqual(v_ea[2], 0.983611496176651 , 6)

    def test_ragf2_outcore(self):
        # tests the out-of-core and chkfile support for AGF2 for H2O/cc-pvdz
        gf2 = agf2.RAGF2(self.mf)
        gf2.max_memory = 1
        gf2.conv_tol = 1e-7
        gf2.run()
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -75.89108074653284 , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.3323976129158455, 6)
        self.assertAlmostEqual(e_ip,     0.4463033078563678 , 6)
        self.assertAlmostEqual(v_ip,     0.9690389634580812 , 6)
        self.assertAlmostEqual(e_ea,     0.1572194156451459 , 6)
        self.assertAlmostEqual(v_ea,     0.9906842311023561 , 6)
        gf2.dump_chk()
        gf2 = agf2.RAGF2(self.mf)
        gf2.__dict__.update(lib.chkfile.load(gf2.chkfile, 'agf2'))
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -75.89108074653284 , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.3323976129158455, 6)
        self.assertAlmostEqual(e_ip,     0.4463033078563678 , 6)
        self.assertAlmostEqual(v_ip,     0.9690389634580812 , 6)
        self.assertAlmostEqual(e_ea,     0.1572194156451459 , 6)
        self.assertAlmostEqual(v_ea,     0.9906842311023561 , 6)


if __name__ == '__main__':
    print('RAGF2 calculations for H2O')
    unittest.main()
