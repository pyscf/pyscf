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
from pyscf import mcscf
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

mol.basis = {'H': '6-31g'}
mol.build()

m = scf.RHF(mol)
m.conv_tol = 1e-15
m.conv_tol_grad = 1e-7
ehf = m.scf()

norb = m.mo_coeff.shape[1]
nelec = mol.nelectron
h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff)).round(9)
g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False).round(9)
na = fci.cistring.num_strings(norb, nelec//2)
numpy.random.seed(15)
ci0 = numpy.random.random((na,na))
ci0 = ci0 + ci0.T
ci0 /= numpy.linalg.norm(ci0)
ci1 = numpy.random.random((na,na))
ci1 = ci1 + ci1.T
ci1 /= numpy.linalg.norm(ci1)

class KnownValues(unittest.TestCase):
    def test_contract(self):
        ci1 = fci.direct_spin0.contract_1e(h1e, ci0, norb, nelec)
        ci1ref = fci.direct_spin1.contract_1e(h1e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 9.1191973750140729, 8)
        ci1 = fci.direct_spin0.contract_2e(g2e, ci0, norb, nelec)
        ci1ref = fci.direct_spin1.contract_2e(g2e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 15.076640155228787, 7)

    def test_kernel(self):
        e, c = fci.direct_spin0.kernel(h1e, g2e, norb, nelec)
        self.assertAlmostEqual(e, -9.1491239851241737, 8)
        e = fci.direct_spin0.energy(h1e, g2e, c, norb, nelec)
        self.assertAlmostEqual(e, -9.1491239851241737, 8)

    def test_rdm1(self):
        dm1ref = fci.direct_spin1.make_rdm1(ci0, norb, nelec)
        dm1 = fci.direct_spin0.make_rdm1(ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.7059849569286722, 10)

    def test_rdm12(self):
        dm1ref, dm2ref = fci.direct_spin1.make_rdm12(ci0, norb, nelec)
        dm1, dm2 = fci.direct_spin0.make_rdm12(ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertTrue(numpy.allclose(dm2ref, dm2))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.7059849569286731, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 7.8811473403497736, 10)

    def test_trans_rdm1(self):
        dm1ref = fci.direct_spin1.trans_rdm1(ci0, ci1, norb, nelec)
        dm1 = fci.direct_spin0.trans_rdm1(ci0, ci1, norb, nelec)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.5485017426647461, 10)
        dm0 = fci.direct_spin0.make_rdm1(ci0, norb, nelec)
        dm1 = fci.direct_spin0.trans_rdm1(ci0, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm1, dm0))

    def test_trans_rdm12(self):
        dm1ref, dm2ref = fci.direct_spin1.trans_rdm12(ci0, ci1, norb, nelec)
        dm1, dm2 = fci.direct_spin0.trans_rdm12(ci0, ci1, norb, nelec)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertTrue(numpy.allclose(dm2ref, dm2))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.5485017426647461, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 7.7327573770739235, 10)
        _,dm0 = fci.direct_spin0.make_rdm12(ci0, norb, nelec)
        _,dm2 = fci.direct_spin0.trans_rdm12(ci0, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm2,dm0))

    def test_davidson_only(self):
        x = 3.0 * 0.529177249
        y = (2.54 - 0.46 * 3.0) * 0.529177249
        mol = gto.M(
            verbose = 0,
            atom = [
            ['Be',( 0., 0.    , 0.   )],
            ['H', ( x, -y    , 0.    )],
            ['H', ( x,  y    , 0.    )],],
            symmetry = True,
            basis = '6-311g')
        mf = scf.RHF(mol)
        mf.scf()
        mf._scf = mf
        h1e = mcscf.casci.h1e_for_cas(mf, mf.mo_coeff, ncas=2, ncore=2)[0]
        eri = ao2mo.incore.full(mf._eri, mf.mo_coeff[:,2:4])
        cis = fci.direct_spin0.FCISolver(mol)
        cis.davidson_only = True
        ci0 = numpy.zeros((2,2))
        ci0[0,0] = 1
        e, c = cis.kernel(h1e, eri, 2, 2, ci0)
        self.assertAlmostEqual(e, -0.80755526695538049, 10)


if __name__ == "__main__":
    print("Full Tests for spin0")
    unittest.main()

