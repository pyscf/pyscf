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
from functools import reduce

from pyscf import gto, scf, ao2mo
from pyscf import cc
from pyscf.cc import ccsd_lambda


class KnownValues(unittest.TestCase):
    def test_ccsd(self):
        mol = gto.M()
        mf = scf.RHF(mol)
        mcc = cc.CCSD(mf)
        numpy.random.seed(12)
        mcc.nocc = nocc = 5
        mcc.nmo = nmo = 12
        nvir = nmo - nocc
        eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
        eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
        fock0 = numpy.random.random((nmo,nmo))
        fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*2
        t1 = numpy.random.random((nocc,nvir))
        t2 = numpy.random.random((nocc,nocc,nvir,nvir))
        t2 = t2 + t2.transpose(1,0,3,2)
        l1 = numpy.random.random((nocc,nvir))
        l2 = numpy.random.random((nocc,nocc,nvir,nvir))
        l2 = l2 + l2.transpose(1,0,3,2)

        eris = cc.ccsd._ChemistsERIs()
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        idx = numpy.tril_indices(nvir)
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:][:,:,idx[0],idx[1]].copy()
        eris.vvvv = ao2mo.restore(4,eri0[nocc:,nocc:,nocc:,nocc:],nvir)
        eris.fock = fock0

        saved = ccsd_lambda.make_intermediates(mcc, t1, t2, eris)
        l1new, l2new = ccsd_lambda.update_lambda(mcc, t1, t2, l1, l2, eris, saved)
        self.assertAlmostEqual(abs(l1new).sum(), 38172.7896467303, 8)
        self.assertAlmostEqual(numpy.dot(l1new.flatten(), numpy.arange(35)), 739312.005491083, 8)
        self.assertAlmostEqual(numpy.dot(l1new.flatten(), numpy.sin(numpy.arange(35))), 7019.50937051188, 8)
        self.assertAlmostEqual(numpy.dot(numpy.sin(l1new.flatten()), numpy.arange(35)), 69.6652346635955, 8)

        self.assertAlmostEqual(abs(l2new).sum(), 72035.4931071527, 8)
        self.assertAlmostEqual(abs(l2new-l2new.transpose(1,0,3,2)).sum(), 0, 9)
        self.assertAlmostEqual(numpy.dot(l2new.flatten(), numpy.arange(35**2)), 48427109.5409886, 7)
        self.assertAlmostEqual(numpy.dot(l2new.flatten(), numpy.sin(numpy.arange(35**2))), 137.758016736487, 8)
        self.assertAlmostEqual(numpy.dot(numpy.sin(l2new.flatten()), numpy.arange(35**2)), 507.656936701192, 8)

        mcc.max_memory = 0
        saved = ccsd_lambda.make_intermediates(mcc, t1, t2, eris)
        l1new, l2new = ccsd_lambda.update_lambda(mcc, t1, t2, l1, l2, eris, saved)
        self.assertAlmostEqual(abs(l1new).sum(), 38172.7896467303, 8)
        self.assertAlmostEqual(numpy.dot(l1new.flatten(), numpy.arange(35)), 739312.005491083, 8)
        self.assertAlmostEqual(numpy.dot(l1new.flatten(), numpy.sin(numpy.arange(35))), 7019.50937051188, 8)
        self.assertAlmostEqual(numpy.dot(numpy.sin(l1new.flatten()), numpy.arange(35)), 69.6652346635955, 8)

        self.assertAlmostEqual(abs(l2new).sum(), 72035.4931071527, 8)
        self.assertAlmostEqual(abs(l2new-l2new.transpose(1,0,3,2)).sum(), 0, 9)
        self.assertAlmostEqual(numpy.dot(l2new.flatten(), numpy.arange(35**2)), 48427109.5409886, 7)
        self.assertAlmostEqual(numpy.dot(l2new.flatten(), numpy.sin(numpy.arange(35**2))), 137.758016736487, 8)
        self.assertAlmostEqual(numpy.dot(numpy.sin(l2new.flatten()), numpy.arange(35**2)), 507.656936701192, 8)

if __name__ == "__main__":
    print("Full Tests for CCSD lambda")
    unittest.main()
