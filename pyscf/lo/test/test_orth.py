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
from pyscf.lo import orth

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = '''
         O    0.   0.       0
         1    0.   -0.757   0.587
         1    0.   0.757    0.587'''

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol)

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_orth(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.random((n,n))
        s = numpy.dot(a.T, a)
        c = orth.lowdin(s)
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(n)))
        x1 = numpy.dot(a, c)
        x2 = orth.vec_lowdin(a)
        d = numpy.dot(x1.T,x2)
        d[numpy.diag_indices(n)] = 0
        self.assertAlmostEqual(numpy.linalg.norm(d), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(c), 36.56738258719514, 9)
        self.assertAlmostEqual(abs(c).sum(), 2655.5580057303964, 7)

    def test_schmidt(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.random((n,n))
        s = numpy.dot(a.T, a)
        c = orth.schmidt(s)
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(n)))
        x1 = numpy.dot(a, c)
        x2 = orth.vec_schmidt(a)
        d = numpy.dot(x1.T,x2)
        d[numpy.diag_indices(n)] = 0
        self.assertAlmostEqual(numpy.linalg.norm(d), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(c), 36.56738258719514, 9)
        self.assertAlmostEqual(abs(c).sum(), 1123.2089785000373, 7)

    def test_weight_orth(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.random((n,n))
        s = numpy.dot(a.T, a)
        weight = numpy.random.random(n)
        c = orth.weight_orth(s, weight)
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(n)))
        self.assertAlmostEqual(numpy.linalg.norm(c), 36.56738258719514, 8)
        self.assertAlmostEqual(abs(c).sum(), 1908.8535852660757, 6)

    def test_orth_ao(self):
        c0 = orth.pre_orth_ao(mol, method='scf')
        self.assertAlmostEqual(abs(c0).sum(), 33.48215772351, 7)
        c = orth.orth_ao(mol, 'lowdin', c0)
        self.assertAlmostEqual(abs(c).sum(), 94.21571091299639, 7)
        c = orth.orth_ao(mol, 'meta_lowdin', c0)
        self.assertAlmostEqual(abs(c).sum(), 92.15697348744733, 7)

        c = orth.orth_ao(mol, 'meta_lowdin', 'sto-3g')
        self.assertAlmostEqual(abs(c).sum(), 90.12324660084619, 7)

        c = orth.orth_ao(mol, 'meta_lowdin', None)
        self.assertAlmostEqual(abs(c).sum(), 83.71349158130113, 7)

    def test_ghost_atm_meta_lowdin(self):
        mol = gto.Mole()
        mol.atom = [["O" , (0. , 0.     , 0.)],
                    ['ghost'   , (0. , -0.757, 0.587)],
                    [1   , (0. , 0.757 , 0.587)] ]
        mol.spin = 1
        mol.basis = {'O':'ccpvdz', 'H':'ccpvdz',
                     'GHOST': gto.basis.load('631g','H')}
        mol.build()
        c = orth.orth_ao(mol, method='meta_lowdin')
        self.assertAlmostEqual(numpy.linalg.norm(c), 7.9067188905237256, 9)

    def test_pre_orth_ao_with_ecp(self):
        mol = gto.M(atom='Cu 0. 0. 0.; H  0.  0. -1.56; H  0.  0.  1.56',
                    basis={'Cu':'lanl2dz', 'H':'ccpvdz'},
                    ecp = {'cu':'lanl2dz'},
                    charge=-1,
                    verbose=0)
        c0 = orth.pre_orth_ao(mol, method='ano')
        self.assertAlmostEqual(numpy.linalg.norm(c0), 5.9621174285790959, 9)


if __name__ == "__main__":
    print("Test orth")
    unittest.main()
