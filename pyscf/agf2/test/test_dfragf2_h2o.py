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
        self.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        self.mf = scf.RHF(self.mol).density_fit(auxbasis='cc-pv5z-ri')
        self.mf.conv_tol = 1e-12
        self.mf.run()
        self.gf2 = agf2.RAGF2(self.mf)
        self.gf2.conv_tol = 1e-7
        self.gf2.run()

    @classmethod
    def tearDownClass(self):
        del self.mol, self.mf, self.gf2

    def test_dfragf2_h2o_ground_state(self):
        # tests the ground state AGF2 energies for H2O/cc-pvdz
        self.assertTrue(self.gf2.converged)
        self.assertAlmostEqual(self.mf.e_tot,  -76.0167894720742   , 4)
        self.assertAlmostEqual(self.gf2.e_1b,  -75.89108074396137  , 4)
        self.assertAlmostEqual(self.gf2.e_2b,  -0.33248785652834784, 4)
        self.assertAlmostEqual(self.gf2.e_init, -0.17330473289845347, 4)

    def test_dfragf2_h2o_ip(self):
        # tests the AGF2 ionization potentials for H2O/cc-pvdz
        e_ip, v_ip = self.gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.45080222600137465, 4)
        self.assertAlmostEqual(e_ip[1], 0.5543195106668687 , 4)
        self.assertAlmostEqual(e_ip[2], 0.6299640547362962 , 4)
        self.assertAlmostEqual(v_ip[0], 0.9704061235804103 , 4)
        self.assertAlmostEqual(v_ip[1], 0.9702372037466642 , 4)
        self.assertAlmostEqual(v_ip[2], 0.9713854565834782 , 4)

    def test_dfragf2_h2o_ea(self):
        # tests the AGF2 electron affinities for H2O/cc-pvdz
        e_ea, v_ea = self.gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.15581330758457984, 4)
        self.assertAlmostEqual(e_ea[1], 0.2347918376963518 , 4)
        self.assertAlmostEqual(e_ea[2], 0.686105303143818  , 4)
        self.assertAlmostEqual(v_ea[0], 0.9903734898112396 , 4)
        self.assertAlmostEqual(v_ea[1], 0.9901410412716749 , 4)
        self.assertAlmostEqual(v_ea[2], 0.9827713231118138 , 4)


if __name__ == '__main__':
    print('DF-RAGF2 calculations for H2O')
    unittest.main()
