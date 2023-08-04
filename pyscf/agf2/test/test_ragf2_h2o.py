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
import tempfile
import numpy as np
import h5py
from pyscf import gto, scf, agf2, lib


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        self.mf = scf.RHF(self.mol)
        self.mf.chkfile = tempfile.NamedTemporaryFile().name
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
        self.assertAlmostEqual(self.gf2.e_1b,  -75.89108074396137  ,  6)
        self.assertAlmostEqual(self.gf2.e_2b,  -0.33248785652834784,  6)
        self.assertAlmostEqual(self.gf2.e_init, -0.17330473289845347,  6)

    def test_ragf2_h2o_ip(self):
        # tests the AGF2 ionization potentials for H2O/cc-pvdz
        e_ip, v_ip = self.gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.45080222600137465, 6)
        self.assertAlmostEqual(e_ip[1], 0.5543195106668687 , 6)
        self.assertAlmostEqual(e_ip[2], 0.6299640547362962 , 6)
        self.assertAlmostEqual(v_ip[0], 0.9704061235804103 , 6)
        self.assertAlmostEqual(v_ip[1], 0.9702372037466642 , 6)
        self.assertAlmostEqual(v_ip[2], 0.9713854565834782 , 6)

    def test_ragf2_h2o_ea(self):
        # tests the AGF2 electron affinities for H2O/cc-pvdz
        e_ea, v_ea = self.gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.15581330758457984, 6)
        self.assertAlmostEqual(e_ea[1], 0.2347918376963518 , 6)
        self.assertAlmostEqual(e_ea[2], 0.686105303143818  , 6)
        self.assertAlmostEqual(v_ea[0], 0.9903734898112396 , 6)
        self.assertAlmostEqual(v_ea[1], 0.9901410412716749 , 6)
        self.assertAlmostEqual(v_ea[2], 0.9827713231118138 , 6)

    def test_ragf2_outcore(self):
        # tests the out-of-core and chkfile support for AGF2 for H2O/cc-pvdz
        gf2 = agf2.RAGF2(self.mf)
        gf2.max_memory = 1
        gf2.incore_complete = False
        gf2.conv_tol = 1e-7
        gf2.run()
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -75.89108074396137  , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.33248785652834784, 6)
        self.assertAlmostEqual(e_ip,     0.45080222600137465 , 6)
        self.assertAlmostEqual(v_ip,     0.9704061235804103  , 6)
        self.assertAlmostEqual(e_ea,     0.15581330758457984 , 6)
        self.assertAlmostEqual(v_ea,     0.9903734898112396  , 6)
        gf2.dump_chk()
        with h5py.File(gf2.chkfile, 'r') as f:
            self.assertEqual(
                set(f['agf2'].keys()),
                {'e_1b', 'e_2b', 'e_init', 'converged', 'mo_energy', 'mo_coeff',
                 'mo_occ', '_nmo', '_nocc', 'se', 'gf'})

        gf2 = agf2.RAGF2(self.mf)
        gf2.__dict__.update(agf2.chkfile.load(gf2.chkfile, 'agf2'))
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -75.89108074396137  , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.33248785652834784, 6)
        self.assertAlmostEqual(e_ip,     0.45080222600137465 , 6)
        self.assertAlmostEqual(v_ip,     0.9704061235804103  , 6)
        self.assertAlmostEqual(e_ea,     0.15581330758457984 , 6)
        self.assertAlmostEqual(v_ea,     0.9903734898112396  , 6)

    def test_ragf2_frozen(self):
        # test the frozen implementation
        mf = scf.RHF(self.mol)
        mf.conv_tol = self.mf.conv_tol
        mf.run()
        gf2 = agf2.RAGF2(mf)
        gf2.conv_tol = 1e-7
        gf2.frozen = [2]
        gf2.run()
        e_ip, v_ip = gf2.ipagf2(nroots=1)
        e_ea, v_ea = gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -75.90803303224045 , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.2378736532302642, 6)
        self.assertAlmostEqual(e_ip,     0.45937490994065694, 6)
        self.assertAlmostEqual(v_ip,     0.9726061540589924 , 6)
        self.assertAlmostEqual(e_ea,     0.15201672352177295, 6)
        self.assertAlmostEqual(v_ea,     0.988560730917133  , 6)

    def test_ragf2_frozen_outcore(self):
        # test the frozen implementation with outcore QMOs
        mf = scf.RHF(self.mol)
        mf.conv_tol = self.mf.conv_tol
        mf.incore_complete = True
        mf.run()
        gf2 = agf2.RAGF2(mf)
        gf2.conv_tol = 1e-7
        gf2.frozen = [2]
        eri = gf2.ao2mo()
        gf2.max_memory = 1
        gf2.incore_complete = False
        gf2.kernel(eri=eri)
        e_ip, v_ip = gf2.ipagf2(nroots=1)
        e_ea, v_ea = gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -75.90803303224045 , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.2378736532302642, 6)
        self.assertAlmostEqual(e_ip,     0.45937490994065694, 6)
        self.assertAlmostEqual(v_ip,     0.9726061540589924 , 6)
        self.assertAlmostEqual(e_ea,     0.15201672352177295, 6)
        self.assertAlmostEqual(v_ea,     0.988560730917133  , 6)

    def test_ragf2_frozen_fully_outcore(self):
        # test the frozen implementation with outcore MOs and QMOs
        mf = scf.RHF(self.mol)
        mf.conv_tol = self.mf.conv_tol
        mf.max_memory = 1
        mf.run()
        gf2 = agf2.RAGF2(mf)
        gf2.conv_tol = 1e-7
        gf2.frozen = [2]
        gf2.kernel()
        e_ip, v_ip = gf2.ipagf2(nroots=1)
        e_ea, v_ea = gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -75.90803303224045 , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.2378736532302642, 6)
        self.assertAlmostEqual(e_ip,     0.45937490994065694, 6)
        self.assertAlmostEqual(v_ip,     0.9726061540589924 , 6)
        self.assertAlmostEqual(e_ea,     0.15201672352177295, 6)
        self.assertAlmostEqual(v_ea,     0.988560730917133  , 6)

if __name__ == '__main__':
    print('RAGF2 calculations for H2O')
    unittest.main()
