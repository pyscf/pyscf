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

import ctypes
import unittest
from functools import reduce
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
          h     0    -0.757   0.587
          h     0    0.757    0.587'''
    mol.basis = 'cc-pvdz'
    mol.build()

def tearDownModule():
    global mol
    del mol

class KnowValues(unittest.TestCase):
    def test_get_ao_eri(self):
        eri = ao2mo.get_ao_eri(gto.M(atom='He'))
        self.assertAlmostEqual(eri[0,0], 1.0557129427350722, 12)

    def test_kernel(self):
        nao = mol.nao
        mo = numpy.random.random((nao,4))
        mo1 = numpy.random.random((nao,2))
        eri = ao2mo.kernel(mol, mo, intor='int2e')
        self.assertEqual(eri.shape, (10,10))

        eri = ao2mo.kernel(mol, [mo,mo1,mo1,mo], intor='int2e')
        self.assertEqual(eri.shape, (8,8))

        eri = ao2mo.kernel(mol, [mo,mo,mo1,mo1])
        self.assertEqual(eri.shape, (10,3))

        eri = ao2mo.kernel(mol, [mo,mo,mo1,mo1])
        self.assertEqual(eri.shape, (10,3))

        nao = mol.nao_2c()
        mo = numpy.random.random((nao,4))
        mo1 = numpy.random.random((nao,2))
        eri = ao2mo.kernel(mol, mo, intor='int2e_spinor')
        self.assertEqual(eri.shape, (16,16))

        eri = ao2mo.kernel(mol, [mo,mo,mo1,mo1], intor='int2e_spinor')
        self.assertEqual(eri.shape, (16,4))

    def test_full(self):
        nao = mol.nao
        mo = numpy.random.random((nao,4))

        eri = mol.intor('int2e', aosym='s4')
        eri = ao2mo.kernel(eri, mo, compact=False)
        self.assertEqual(eri.shape, (16,16))

        h5file = lib.H5TmpFile()
        ao2mo.kernel(mol, mo, erifile=h5file, intor='int2e', dataname='eri')
        with ao2mo.load(h5file, 'eri') as eri:
            self.assertEqual(eri.shape, (10,10))

        ftmp = tempfile.NamedTemporaryFile()
        ao2mo.kernel(mol, mo, ftmp, intor='int2e', dataname='eri')
        with ao2mo.load(ftmp, 'eri') as eri:
            self.assertEqual(eri.shape, (10,10))

    def test_general(self):
        nao = mol.nao
        mo = numpy.random.random((nao,4))

        eri = mol.intor('int2e', aosym='s8')
        eri = ao2mo.kernel(eri, [mo]*4, compact=False)
        self.assertEqual(eri.shape, (16,16))

        h5file = lib.H5TmpFile()
        ao2mo.kernel(mol, [mo]*4, erifile=h5file, intor='int2e', dataname='eri')
        self.assertEqual(h5file['eri'].shape, (10,10))

        ftmp = tempfile.NamedTemporaryFile()
        ao2mo.kernel(mol, [mo]*4, ftmp, intor='int2e', dataname='eri')
        with ao2mo.load(ftmp.name, 'eri') as eri:
            self.assertEqual(eri.shape, (10,10))


if __name__ == '__main__':
    print('Full Tests for __init__')
    unittest.main()
