#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
import numpy
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.cc import rccsd

from pyscf.cc import ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm
from pyscf.cc import ccsd_t_lambda_slow
from pyscf.cc import ccsd_t_rdm_slow


def setUpModule():
    global mol, mf, mycc
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.build()
    mf = scf.RHF(mol).run()
    mycc = rccsd.RCCSD(mf)

def tearDownModule():
    global mol, mf, mycc
    mol.stdout.close()
    del mol, mf, mycc

class KnownValues(unittest.TestCase):
    def test_lambda_intermediates_real(self):
        mycc = rccsd.RCCSD(mf)
        np.random.seed(12)
        nocc = 5
        nmo = 12
        nvir = nmo - nocc
        eri0 = np.random.random((nmo,nmo,nmo,nmo))
        eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
        fock0 = np.random.random((nmo,nmo))
        fock0 = fock0 + fock0.T + np.diag(range(nmo))*2
        t1 = np.random.random((nocc,nvir))
        t2 = np.random.random((nocc,nocc,nvir,nvir))
        t2 = t2 + t2.transpose(1,0,3,2)

        eris = rccsd._ChemistsERIs(mol)
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        idx = np.tril_indices(nvir)
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        eris.fock = fock0
        eris.mo_energy = fock0.diagonal()

        imds_ref = ccsd_t_lambda_slow.make_intermediates(mycc, t1, t2, eris)

        imds = ccsd_t_lambda.make_intermediates(mycc, t1, t2, eris)
        self.assertAlmostEqual(lib.fp(imds.l1_t), 19.765863433294747, 9)
        self.assertAlmostEqual(lib.fp(imds.l2_t), -5.307912647007477, 9)
        self.assertAlmostEqual(abs(imds.l2_t-imds.l2_t.transpose(1,0,3,2)).max(), 0, 12)
        self.assertTrue(np.allclose(imds.l1_t, imds_ref.l1_t, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(imds.l2_t, imds_ref.l2_t, rtol=1e-12, atol=1e-15))

        mycc.max_memory = 118
        imds = ccsd_t_lambda.make_intermediates(mycc, t1, t2, eris)
        self.assertAlmostEqual(lib.fp(imds.l1_t), 19.765863433294747, 9)
        self.assertAlmostEqual(lib.fp(imds.l2_t), -5.307912647007477, 9)
        self.assertAlmostEqual(abs(imds.l2_t-imds.l2_t.transpose(1,0,3,2)).max(), 0, 12)
        self.assertTrue(np.allclose(imds.l1_t, imds_ref.l1_t, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(imds.l2_t, imds_ref.l2_t, rtol=1e-12, atol=1e-15))

        mycc.max_memory = 0
        imds = ccsd_t_lambda.make_intermediates(mycc, t1, t2, eris)
        self.assertAlmostEqual(lib.fp(imds.l1_t), 19.765863433294747, 9)
        self.assertAlmostEqual(lib.fp(imds.l2_t), -5.307912647007477, 9)
        self.assertAlmostEqual(abs(imds.l2_t-imds.l2_t.transpose(1,0,3,2)).max(), 0, 12)
        self.assertTrue(np.allclose(imds.l1_t, imds_ref.l1_t, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(imds.l2_t, imds_ref.l2_t, rtol=1e-12, atol=1e-15))

    def test_rdm_intermediates_real(self):
        mycc = rccsd.RCCSD(mf)
        np.random.seed(12)
        nocc = 5
        nmo = 12
        nvir = nmo - nocc
        eri0 = np.random.random((nmo,nmo,nmo,nmo))
        eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
        fock0 = np.random.random((nmo,nmo))
        fock0 = fock0 + fock0.T + np.diag(range(nmo))*2
        t1 = np.random.random((nocc,nvir))
        t2 = np.random.random((nocc,nocc,nvir,nvir))
        t2 = t2 + t2.transpose(1,0,3,2)
        l1 = np.random.random((nocc,nvir))
        l2 = np.random.random((nocc,nocc,nvir,nvir))
        l2 = l2 + l2.transpose(1,0,3,2)

        eris = rccsd._ChemistsERIs(mol)
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        idx = np.tril_indices(nvir)
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        eris.fock = fock0
        eris.mo_energy = fock0.diagonal()

        d1_ref = ccsd_t_rdm_slow._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=False)
        d1_ref = np.concatenate([d.ravel() for d in d1_ref])
        d1_g_ref = ccsd_t_rdm_slow._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=True)
        d1_g_ref = np.concatenate([d.ravel() for d in d1_g_ref])
        d2_ref = ccsd_t_rdm_slow._gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv=False)
        d2_ref = np.concatenate([d.ravel() for d in d2_ref])

        d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=False)
        d1 = np.concatenate([d.ravel() for d in d1])
        self.assertAlmostEqual(lib.fp(d1), -1383.6462141528182, 9)
        d1_g = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=True)
        d1_g = np.concatenate([d.ravel() for d in d1_g])
        self.assertAlmostEqual(lib.fp(d1_g), -1305.4979081481401, 9)
        d2 = ccsd_t_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv=False)
        d2 = np.concatenate([d.ravel() for d in d2])
        self.assertAlmostEqual(lib.fp(d2), 13025.265198471607, 9)
        self.assertTrue(np.allclose(d1, d1_ref, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(d1_g, d1_g_ref, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(d2, d2_ref, rtol=1e-12, atol=1e-15))

        mycc.max_memory = 118
        d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=False)
        d1 = np.concatenate([d.ravel() for d in d1])
        self.assertAlmostEqual(lib.fp(d1), -1383.6462141528182, 9)
        d1_g = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=True)
        d1_g = np.concatenate([d.ravel() for d in d1_g])
        self.assertAlmostEqual(lib.fp(d1_g), -1305.4979081481401, 9)
        d2 = ccsd_t_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv=False)
        d2 = np.concatenate([d.ravel() for d in d2])
        self.assertAlmostEqual(lib.fp(d2), 13025.265198471607, 9)
        self.assertTrue(np.allclose(d1, d1_ref, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(d1_g, d1_g_ref, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(d2, d2_ref, rtol=1e-12, atol=1e-15))

        mycc.max_memory = 0
        d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=False)
        d1 = np.concatenate([d.ravel() for d in d1])
        self.assertAlmostEqual(lib.fp(d1), -1383.6462141528182, 9)
        d1_g = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=True)
        d1_g = np.concatenate([d.ravel() for d in d1_g])
        self.assertAlmostEqual(lib.fp(d1_g), -1305.4979081481401, 9)
        d2 = ccsd_t_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv=False)
        d2 = np.concatenate([d.ravel() for d in d2])
        self.assertAlmostEqual(lib.fp(d2), 13025.265198471607, 9)
        self.assertTrue(np.allclose(d1, d1_ref, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(d1_g, d1_g_ref, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.allclose(d2, d2_ref, rtol=1e-12, atol=1e-15))

if __name__ == "__main__":
    print("Tests for RCCSD(T) lambda and rdm intermediates")
    unittest.main()
