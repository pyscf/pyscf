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
from pyscf import gto
from pyscf.scf import _response_functions

class KnownValues(unittest.TestCase):
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

    def _check_dm0_response(self, mf, dm1, with_nlc):
        dm0 = mf.make_rdm1()
        vind_mo = mf.gen_response(mf.mo_coeff, mf.mo_occ, with_nlc=with_nlc)
        vind_dm0 = mf.gen_response(dm0=dm0, with_nlc=with_nlc)
        self.assertAlmostEqual(abs(vind_mo(dm1) - vind_dm0(dm1)).max(), 0, 10)

    def test_rhf_response_dm0(self):
        mol = gto.M(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            basis = '631g')
        nao = mol.nao
        dm1 = np.random.rand(nao, nao)
        dm1 = dm1 + dm1.T

        mf = mol.RHF().run()
        self._check_dm0_response(mf, dm1, with_nlc=True)

        mf = mol.RKS(xc='wb97mv').run()
        mf.nlcgrids.level = 0
        self._check_dm0_response(mf, dm1, with_nlc=True)

    def test_uhf_response_dm0(self):
        mol = gto.M(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            charge = 1,
            spin = 1,
            basis = '631g')
        nao = mol.nao
        dm1 = np.random.rand(2, nao, nao)
        dm1 = dm1 + dm1.transpose(0, 2, 1)

        mf = mol.UHF().run()
        self._check_dm0_response(mf, dm1, with_nlc=True)

        mf = mol.UKS(xc='wb97mv').run()
        mf.nlcgrids.level = 0
        self._check_dm0_response(mf, dm1, with_nlc=True)

    def test_ghf_response_dm0(self):
        mol = gto.M(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            charge = 1,
            spin = 1,
            basis = '631g')

        mf = mol.GHF().run()
        n2c = mf.mo_coeff.shape[0]
        dm1 = np.random.rand(n2c, n2c)
        dm1 = dm1 + dm1.T
        self._check_dm0_response(mf, dm1, with_nlc=True)

        mf = mol.GKS(xc='wb97mv').run()
        mf.nlcgrids.level = 0
        n2c = mf.mo_coeff.shape[0]
        dm1 = np.random.rand(n2c, n2c)
        dm1 = dm1 + dm1.T
        self._check_dm0_response(mf, dm1, with_nlc=True)

    def test_dhf_response_dm0(self):
        mol = gto.M(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            charge = 1,
            spin = 1,
            basis = '631g')

        mf = mol.DHF().run()
        n4c = mf.mo_coeff.shape[0]
        dm1 = np.random.rand(n4c, n4c) + 1j * np.random.rand(n4c, n4c)
        dm1 = dm1 + dm1.conj().T
        self._check_dm0_response(mf, dm1, with_nlc=True)

if __name__ == "__main__":
    print("Full Tests for response_functions")
    unittest.main()
