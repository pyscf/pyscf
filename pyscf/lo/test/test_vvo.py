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
from pyscf import lib
from pyscf import scf
from pyscf.lo import iao, ibo, orth, pipek, vvo

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.atom = '''
         O    0.   0.       0
         h    0.   -0.757   0.587
         h    0.   0.757    0.587'''
    mol.basis = 'unc-sto3g'
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_vvo(self):
        mf = scf.RHF(mol).run()
        nocc = numpy.sum(mf.mo_occ>0)
        b = vvo.vvo(mol, mf.mo_coeff[:,0:nocc], mf.mo_coeff[:,nocc:])
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 0.6695907625196215, 5)

    def test_livvo(self):
        mf = scf.RHF(mol).run()
        nocc = numpy.sum(mf.mo_occ>0)
        b = vvo.livvo(mol, mf.mo_coeff[:,0:nocc], mf.mo_coeff[:,nocc:], exponent=4)
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 1.073138251815934, 5)

        b = vvo.livvo(mol, mf.mo_coeff[:,0:nocc], mf.mo_coeff[:,nocc:], exponent=2)
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 1.073138251815934, 5)

    def test_livvo_PM(self):
        mf = scf.RHF(mol).run()
        nocc = numpy.sum(mf.mo_occ>0)
        b = vvo.livvo(mol, mf.mo_coeff[:,0:nocc], mf.mo_coeff[:,nocc:], locmethod='PM', exponent=4).kernel()
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 0.6695907625196215, 5)

        b = vvo.livvo(mol, mf.mo_coeff[:,0:nocc], mf.mo_coeff[:,nocc:], locmethod='PM', exponent=2).kernel()
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 0.6695907625196215, 5)


if __name__ == "__main__":
    print("Full tests for vvo")
    unittest.main()
