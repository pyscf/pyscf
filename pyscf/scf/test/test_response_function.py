#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import scipy.linalg
from pyscf import lib
from pyscf import dft, gto
from pyscf.scf import _response_functions

class KnownValues(unittest.TestCase):
    def test_rks_second_grids(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis = '631g')
        mf = mol.RKS(xc='b3lyp').run(conv_tol=1e-11)

        td0 = mf.TDA()
        td0.nstates = 3
        e_ref = td0.kernel()[0]

        # By default second_grids is None and response functions use mf.grids
        self.assertIsNone(mf.second_grids)

        # An unbuilt copy of mf.grids gives the same response
        grids_copy = dft.gen_grid.Grids(mol)
        grids_copy.level = mf.grids.level
        grids_copy.prune = mf.grids.prune
        mf.second_grids = grids_copy
        td1 = mf.TDA()
        td1.nstates = 3
        self.assertAlmostEqual(abs(td1.kernel()[0] - e_ref).max(), 0, 9)
        mf.second_grids = None

        # A level-1 secondary grid gives very similar excitations
        mf.set_second_grids(1)
        td2 = mf.TDA()
        td2.nstates = 3
        e2 = td2.kernel()[0]
        self.assertTrue(mf.second_grids.coords.shape[0] < mf.grids.coords.shape[0])
        self.assertAlmostEqual(abs(e2 - e_ref).max(), 0, 4)
        mf.second_grids = None

        # The sg1 scheme builds an SG1 grid
        mf.set_second_grids('sg1')
        self.assertEqual(mf.second_grids.prune, dft.gen_grid.sg1_prune)
        self.assertEqual(mf.second_grids.atom_grid, (50, 194))
        td3 = mf.TDA()
        td3.nstates = 3
        e3 = td3.kernel()[0]
        self.assertAlmostEqual(abs(e3 - e_ref).max(), 0, 4)
        mf.second_grids = None

    def test_uks_second_grids_vind(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis = '631g')
        mf = mol.UKS(xc='pbe').run()
        nao = mol.nao
        np.random.seed(1)
        dm1 = np.random.rand(2, nao, nao)

        mf.set_second_grids(1)
        second_grids = mf.second_grids
        v1 = mf.gen_response()(dm1)
        mf.second_grids = None
        # The grids kwarg of gen_response bypasses mf.second_grids
        v2 = mf.gen_response(grids=second_grids)(dm1)
        self.assertAlmostEqual(abs(v1 - v2).max(), 0, 12)
        v3 = mf.gen_response(grids=mf.grids)(dm1)
        v4 = mf.gen_response()(dm1)
        self.assertAlmostEqual(abs(v3 - v4).max(), 0, 12)

    def test_gks_nlc(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            charge = 1,
            spin = 1,
            basis = '631g')
        nao = mol.nao
        mf_uks = mol.UHF().run().to_uks()
        mf_uks.xc = 'wb97mv'
        mf_uks.nlcgrids.level = 0

        dm = mf_uks.make_rdm1()
        dm1 = np.random.rand(2, nao, nao)
        vind = mf_uks.gen_response(with_nlc=True)
        ref = scipy.linalg.block_diag(*vind(dm1))

        mf_gks = mf_uks.to_gks()
        vind = mf_gks.gen_response(with_nlc=True)
        v = vind(scipy.linalg.block_diag(*dm1))
        self.assertAlmostEqual(abs(v - ref).max(), 0, 12)

if __name__ == "__main__":
    print("Full Tests for response_functions")
    unittest.main()
