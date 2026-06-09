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
    
    def test_pre_orth_ao_with_coreless_ecp(self):
        mol = gto.M(atom = '''
            ghost-Cu2  2.33770     1.38257    -2.24106
            Cu0       -0.51523     1.42830    -3.17698
            Cu0        0.39700     3.33838    -1.04703
            Cu0        1.33168     3.68956    -3.87904
            O0         0.88779     2.45970    -2.58603
            O0         3.78761     0.30544    -1.89609
            O0        -0.09379     4.21706     0.49196
            O0        -1.91824     0.39690    -3.76793
            O0         1.77557     4.91941    -5.17205
            X-Cu1      5.19063     1.33684    -1.30514
            X-Cu1      4.27840    -0.57324    -3.43509
            X-Cu1      3.34372    -0.92441    -0.60308
            X-Cu1      1.30922     5.24846     1.08291
            X-Cu1      3.17859     5.95081    -4.58110
            X-Cu1     -1.42745    -0.48178    -5.30692
            X-Cu1      2.26636     4.04073    -6.71105
            X-Cu1     -0.53769     2.98721     1.78498
            X-Cu1     -2.36214    -0.83295    -2.47491
            X-Cu1      0.32566     5.99654    -5.51702
            X-Cu1     -1.54371     5.29420     0.14700
            X-Cu1     -3.36816     1.47403    -4.11289
            ''',
            verbose=0,
            spin=1,
            charge=-6,
            basis={'Cu0': 'cc-pvdz', 'O0': 'cc-pvdz', 'ghost-Cu2': gto.basis.load('cc-pvdz', 'Cu')},
            ecp={'Cu': 'cc-pvdz-pp',
                    'X-Cu1': gto.basis.parse_ecp('''
                    Cu nelec 0
                    Cu ul
                    2       1.000000000            0.000000000
                    Cu S
                    2      30.220000000          355.770158000
                    2      13.190000000           70.865357000
                    Cu P
                    2      33.130000000          233.891976000
                    2      13.220000000           53.947299000
                    Cu D
                    2      38.420000000          -31.272165000
                    2      13.260000000           -2.741104000
                    ''')})

        c = orth.pre_orth_ao(mol, method='ano')
        self.assertAlmostEqual(numpy.linalg.norm(c), 35.14205617894, 9)

if __name__ == "__main__":
    print("Test orth")
    unittest.main()
