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
from pyscf.lo import iao, ibo, orth, pipek

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
    def test_ibo(self):
        mf = scf.RHF(mol).run()
        b = ibo.ibo(mol, mf.mo_coeff[:,mf.mo_occ>0], exponent=4)
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 4.0661421502005437, 4)

        b = ibo.ibo(mol, mf.mo_coeff[:,mf.mo_occ>0], exponent=2)
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 4.0661421502005437, 4)

    def test_ibo_PM(self):
        mf = scf.RHF(mol).run()
        b = ibo.ibo(mol, mf.mo_coeff[:,mf.mo_occ>0], locmethod='PM', exponent=4).kernel()
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 3.9201797890974261, 4)

        b = ibo.ibo(mol, mf.mo_coeff[:,mf.mo_occ>0], locmethod='PM', exponent=2).kernel()
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 3.9201797890974261, 4)


if __name__ == "__main__":
    print("Full tests for ibo")
    unittest.main()
