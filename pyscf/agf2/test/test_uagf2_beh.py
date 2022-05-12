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
from pyscf import gto, scf, agf2, lib


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.mol = gto.M(atom='Be 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        self.mf = scf.UHF(self.mol)
        self.mf.chkfile = tempfile.NamedTemporaryFile().name
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
        self.assertAlmostEqual(self.gf2.e_1b,  -15.069681001221705  ,  6)
        self.assertAlmostEqual(self.gf2.e_2b,  -0.049461593728309786,  6)
        self.assertAlmostEqual(self.gf2.e_init, -0.025198374705580943,  6)

    def test_uagf2_beh_ip(self):
        # tests the AGF2 ionization potentials for BeH/cc-pvdz
        e_ip, v_ip = self.gf2.ipagf2(nroots=3)
        v_ip = [np.linalg.norm(v)**2 for v in v_ip]
        self.assertAlmostEqual(e_ip[0], 0.3003522286132736, 6)
        self.assertAlmostEqual(e_ip[1], 0.5107596660196604, 6)
        self.assertAlmostEqual(e_ip[2], 0.5318094633979558, 6)
        self.assertAlmostEqual(v_ip[0], 0.9962231685493768, 6)
        self.assertAlmostEqual(v_ip[1], 0.9789822411853315, 6)
        self.assertAlmostEqual(v_ip[2], 0.9809062972345126, 6)

    def test_uagf2_beh_ea(self):
        # tests the AGF2 electron affinities for BeH/cc-pvdz
        e_ea, v_ea = self.gf2.eaagf2(nroots=3)
        v_ea = [np.linalg.norm(v)**2 for v in v_ea]
        self.assertAlmostEqual(e_ea[0], 0.03781071654337435, 6)
        self.assertAlmostEqual(e_ea[1], 0.04252189700736402, 6)
        self.assertAlmostEqual(e_ea[2], 0.0425218970073656 , 6)
        self.assertAlmostEqual(v_ea[0], 0.9740024912068087 , 6)
        self.assertAlmostEqual(v_ea[1], 0.9902310149008003 , 6)
        self.assertAlmostEqual(v_ea[2], 0.9902310149008006 , 6)

    def test_uagf2_outcore(self):
        # tests the out-of-core and chkfile support for AGF2 for BeH/cc-pvdz
        gf2 = agf2.UAGF2(self.mf)
        gf2.chkfile = tempfile.NamedTemporaryFile().name
        gf2.max_memory = 1
        gf2.conv_tol = 1e-7
        gf2.run()
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(self.gf2.e_1b, -15.069681001221705  , 6)
        self.assertAlmostEqual(self.gf2.e_2b, -0.049461593728309786, 6)
        self.assertAlmostEqual(e_ip,          0.3003522286132736   , 6)
        self.assertAlmostEqual(v_ip,          0.9962231685493768   , 6)
        self.assertAlmostEqual(e_ea,          0.03781071654337435  , 6)
        self.assertAlmostEqual(v_ea,          0.9740024912068087   , 6)
        gf2.dump_chk()
        gf2 = agf2.UAGF2(self.mf)
        gf2.__dict__.update(agf2.chkfile.load(gf2.chkfile, 'agf2'))
        e_ip, v_ip = self.gf2.ipagf2(nroots=1)
        e_ea, v_ea = self.gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(self.gf2.e_1b, -15.069681001221705  , 6)
        self.assertAlmostEqual(self.gf2.e_2b, -0.049461593728309786, 6)
        self.assertAlmostEqual(e_ip,          0.3003522286132736   , 6)
        self.assertAlmostEqual(v_ip,          0.9962231685493768   , 6)
        self.assertAlmostEqual(e_ea,          0.03781071654337435  , 6)
        self.assertAlmostEqual(v_ea,          0.9740024912068087   , 6)

    def test_uagf2_frozen_outcore(self):
        # test the frozen implementation with outcore QMOs
        mf = scf.UHF(self.mol)
        mf.conv_tol = self.mf.conv_tol
        mf.incore_complete = True
        mf.run()
        gf2 = agf2.UAGF2(mf)
        gf2.conv_tol = 1e-7
        gf2.frozen = [[1], []]
        eri = gf2.ao2mo()
        gf2.max_memory = 1
        gf2.incore_complete = False
        gf2.run(eri=eri)
        e_ip, v_ip = gf2.ipagf2(nroots=1)
        e_ea, v_ea = gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -15.074739470012476  , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.02869128199243178 , 6)
        self.assertAlmostEqual(e_ip,     0.30429793411749156  , 6)
        self.assertAlmostEqual(v_ip,     0.994829573647208    , 6)
        self.assertAlmostEqual(e_ea,     -0.006073553177668935, 6)
        self.assertAlmostEqual(v_ea,     0.8287871847529051   , 6)

    def test_uagf2_frozen_fully_outcore(self):
        # test the frozen implementation with outcore MOs and QMOs
        mf = scf.UHF(self.mol)
        mf.conv_tol = self.mf.conv_tol
        mf.max_memory = 1
        mf.run()
        gf2 = agf2.UAGF2(mf)
        gf2.conv_tol = 1e-7
        gf2.frozen = [[1], []]
        gf2.run()
        e_ip, v_ip = gf2.ipagf2(nroots=1)
        e_ea, v_ea = gf2.eaagf2(nroots=1)
        v_ip = np.linalg.norm(v_ip)**2
        v_ea = np.linalg.norm(v_ea)**2
        self.assertAlmostEqual(gf2.e_1b, -15.074739470012476  , 6)
        self.assertAlmostEqual(gf2.e_2b, -0.02869128199243178 , 6)
        self.assertAlmostEqual(e_ip,     0.30429793411749156  , 6)
        self.assertAlmostEqual(v_ip,     0.994829573647208    , 6)
        self.assertAlmostEqual(e_ea,     -0.006073553177668935, 6)
        self.assertAlmostEqual(v_ea,     0.8287871847529051   , 6)


if __name__ == '__main__':
    print('UAGF2 calculations for BeH')
    unittest.main()
