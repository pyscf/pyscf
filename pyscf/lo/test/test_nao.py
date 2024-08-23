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
from pyscf.lo import nao

def setUpModule():
    global mol, mf, mol1, mf1
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
    mf.conv_tol = 1e-14
    mf.scf()

    mol1 = mol.copy()
    mol1.cart = True
    mf1 = scf.RHF(mol1).set(conv_tol=1e-14).run()

def tearDownModule():
    global mol, mf, mol1, mf1
    del mol, mf, mol1, mf1

class KnownValues(unittest.TestCase):
    def test_pre_nao(self):
        c = nao.prenao(mol, mf.make_rdm1())
        self.assertAlmostEqual(numpy.linalg.norm(c), 5.7742626195362039, 9)
        self.assertAlmostEqual(abs(c).sum(), 33.214804163888289, 6)

        c = nao.prenao(mol1, mf1.make_rdm1())
        self.assertAlmostEqual(abs(c).sum(), 34.21520681, 6)

    def test_nao(self):
        c = nao.nao(mol, mf)
        s = mf.get_ovlp()
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(s.shape[0])))
        self.assertAlmostEqual(numpy.linalg.norm(c), 8.982385484322208, 9)
        self.assertAlmostEqual(abs(c).sum(), 90.443872916389637, 6)

        c = nao.nao(mol1, mf1)
        s = mf1.get_ovlp()
        self.assertAlmostEqual(abs(c.T.dot(s).dot(c) - numpy.eye(s.shape[0])).max(), 0, 8)
        self.assertAlmostEqual(abs(c).sum(), 100.54615191, 6)

    # issue 1755
    def test_nao_pop_symmetry(self):
        mol = gto.M(atom='''
        6 -0.000000 -1.399560 0.000000
        6 -1.212054 -0.699779 -0.000000
        6 1.212054 -0.699779 -0.000000
        1 -2.156901 -1.245293 0.000000
        1 2.156901 -1.245293 0.000000
        6 -1.212054 0.699779 0.000000
        6 1.212054 0.699779 0.000000
        1 -2.156901 1.245293 -0.000000
        1 2.156901 1.245293 -0.000000
        6 -0.000000 1.399560 -0.000000
        1 0.000000 2.490576 0.000000
        1 0.000000 -2.490576 -0.000000
        ''', basis='sto3g', verbose=0)
        mf = scf.RHF(mol).run()
        C = nao.nao(mol, mf)
        mo = C.T.dot(mf.get_ovlp()).dot(mf.mo_coeff)
        dm = mf.make_rdm1(mo, mf.mo_occ)
        pop, chg = mf.mulliken_pop(mol, dm, numpy.eye(mol.nao_nr()), verbose=0)
        self.assertAlmostEqual(abs(abs(chg) - 0.03668).max(), 0, 6)


if __name__ == "__main__":
    print("Test orth")
    unittest.main()
