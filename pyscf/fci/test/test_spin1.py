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
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_h2o"
mol.atom = [
    ['H', ( 1.,-1.    , 0.   )],
    ['H', ( 0.,-1.    ,-1.   )],
    ['H', ( 0.,-0.5   ,-0.   )],
    ['H', ( 0.,-0.    ,-1.   )],
    ['H', ( 1.,-0.5   , 0.   )],
    ['H', ( 0., 1.    , 1.   )],
]

mol.basis = {'H': 'sto-3g'}
mol.build()

m = scf.RHF(mol)
m.conv_tol = 1e-15
ehf = m.scf()

norb = m.mo_coeff.shape[1]
nelec = (mol.nelectron//2, mol.nelectron//2)
h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
na = fci.cistring.num_strings(norb, nelec[0])
nb = fci.cistring.num_strings(norb, nelec[1])
numpy.random.seed(15)
ci0 = numpy.random.random((na,nb))
ci0 = (ci0 + ci0.T) * .5
ci1 = numpy.random.random((na,nb))

neleci = (mol.nelectron//2, mol.nelectron//2-1)
na = fci.cistring.num_strings(norb, neleci[0])
nb = fci.cistring.num_strings(norb, neleci[1])
numpy.random.seed(15)
ci2 = numpy.random.random((na,nb))
ci3 = numpy.random.random((na,nb))

class KnownValues(unittest.TestCase):
    def test_contract(self):
        ci1ref = fci.direct_spin0.contract_1e(h1e, ci0, norb, mol.nelectron)
        ci1 = fci.direct_spin1.contract_1e(h1e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1, ci1ref))
        ci1ref = fci.direct_spin0.contract_2e(g2e, ci0, norb, mol.nelectron)
        ci1 = fci.direct_spin1.contract_2e(g2e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1, ci1ref))
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 195.61063639809828, 6)
        ci3 = fci.direct_spin1.contract_2e(g2e, ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(ci3), 127.49780293866368, 6)

    def test_kernel(self):
        eref, cref = fci.direct_spin0.kernel(h1e, g2e, norb, mol.nelectron)
        e, c = fci.direct_spin1.kernel(h1e, g2e, norb, nelec)
        self.assertAlmostEqual(e, eref, 8)
        self.assertAlmostEqual(e, -8.9347029192929, 8)
        e = fci.direct_spin1.energy(h1e, g2e, c, norb, nelec)
        self.assertAlmostEqual(e, -8.9347029192929, 8)
        e, c = fci.direct_spin1.kernel(h1e, g2e, norb, neleci)
        self.assertAlmostEqual(e, -8.7498253981782, 8)

    def test_hdiag(self):
        hdiagref = fci.direct_spin0.make_hdiag(h1e, g2e, norb, mol.nelectron)
        hdiag = fci.direct_spin1.make_hdiag(h1e, g2e, norb, nelec)
        self.assertTrue(numpy.allclose(hdiag, hdiagref))
        self.assertAlmostEqual(numpy.linalg.norm(hdiag), 133.95118651178, 9)
        hdiag = fci.direct_spin1.make_hdiag(h1e, g2e, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(hdiag), 113.85080162587, 9)

    def test_rdm1(self):
        dm1ref = fci.direct_spin0.make_rdm1(ci0, norb, mol.nelectron)
        dm1 = fci.direct_spin1.make_rdm1(ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 358.89058866411972, 10)
        dm1 = fci.direct_spin1.make_rdm1(ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 242.33237916212, 10)

    def test_rdm12(self):
        dm1ref, dm2ref = fci.direct_spin0.make_rdm12(ci0, norb, mol.nelectron)
        dm1, dm2 = fci.direct_spin1.make_rdm12s(ci0, norb, nelec)
        dm1 = dm1[0] + dm1[1]
        dm2 = dm2[0] + dm2[1] + dm2[1].transpose(2,3,0,1) + dm2[2]
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertTrue(numpy.allclose(dm2ref, dm2))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 358.89058866411972, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 1098.6108689470402, 10)
        dm1, dm2 = fci.direct_spin1.make_rdm12(ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 242.33237916212, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 581.11055963403, 10)

    def test_trans_rdm1(self):
        dm1ref = fci.direct_spin0.trans_rdm1(ci0, ci1, norb, mol.nelectron)
        dm1 = fci.direct_spin1.trans_rdm1(ci0, ci1, norb, nelec)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 298.99106891067782, 10)
        dm0 = fci.direct_spin1.make_rdm1(ci0, norb, nelec)
        dm1 = fci.direct_spin1.trans_rdm1(ci0, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm1, dm0))
        dm1 = fci.direct_spin1.trans_rdm1(ci3, ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 193.703051323676, 10)

    def test_trans_rdm12(self):
        dm1ref, dm2ref = fci.direct_spin0.trans_rdm12(ci0, ci1, norb, mol.nelectron)
        dm1, dm2 = fci.direct_spin1.trans_rdm12s(ci0, ci1, norb, nelec)
        dm1 = dm1[0] + dm1[1]
        dm2 = dm2[0] + dm2[1] + dm2[2] + dm2[3]
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertTrue(numpy.allclose(dm2ref, dm2))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 298.99106891067782, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 959.23757961147089, 10)
        _,dm0 = fci.direct_spin1.make_rdm12(ci0, norb, nelec)
        _,dm2 = fci.direct_spin1.trans_rdm12(ci0, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm2, dm0))
        dm1, dm2 = fci.direct_spin1.trans_rdm12(ci3, ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 193.703051323676, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 512.111790469461, 10)


if __name__ == "__main__":
    print("Full Tests for spin1")
    unittest.main()

