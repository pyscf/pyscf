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
    global mol, nao, eri
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
          o     0    0.       0
          h     0    -0.757   0.587
          h     0    0.757    0.587'''
    mol.basis = 'cc-pvdz'
    mol.build()
    nao = mol.nao_nr()
    eri = mol.intor('int2e_sph', aosym='s8')

def tearDownModule():
    global mol, eri
    del mol, eri

def trans(eri, mos):
    nao = mos[0].shape[0]
    eriref = ao2mo.restore(1, eri, nao)
    eriref = lib.einsum('pjkl,pi->ijkl', eriref, mos[0].conj())
    eriref = lib.einsum('ipkl,pj->ijkl', eriref, mos[1])
    eriref = lib.einsum('ijpl,pk->ijkl', eriref, mos[2].conj())
    eriref = lib.einsum('ijkp,pl->ijkl', eriref, mos[3])
    return eriref

class KnownValues(unittest.TestCase):
    def test_incore(self):
        numpy.random.seed(15)
        nmo = 12
        mo = numpy.random.random((nao,nmo))
        eriref = trans(eri, [mo]*4)

        eri1 = ao2mo.incore.full(ao2mo.restore(8,eri,nao), mo)
        self.assertTrue(numpy.allclose(ao2mo.restore(1,eri1,nmo), eriref))
        eri1 = ao2mo.incore.full(ao2mo.restore(4,eri,nao), mo, compact=False)
        self.assertTrue(numpy.allclose(eri1.reshape((nmo,)*4), eriref))

        eri1 = ao2mo.incore.general(eri, (mo[:,:2], mo[:,1:3], mo[:,:3], mo[:,2:5]))
        eri1 = eri1.reshape(2,2,3,3)
        self.assertTrue(numpy.allclose(eri1, eriref[:2,1:3,:3,2:5]))

#        eri_ao = ao2mo.restore('s2ij', eri, nao)
#        eri1 = ao2mo.incore.general(eri_ao, (mo[:,:3], mo[:,1:3], mo[:,:3], mo[:,2:5]))
#        eri1 = eri1.reshape(3,2,3,3)
#        self.assertTrue(numpy.allclose(eri1, eriref[:3,1:3,:3,2:5]))

        eri_ao = ao2mo.restore(1, eri, nao)
        eri1 = ao2mo.incore.general(eri_ao, (mo[:,:3], mo[:,1:3], mo[:,:3], mo[:,2:5]))
        eri1 = eri1.reshape(3,2,3,3)
        self.assertTrue(numpy.allclose(eri1, eriref[:3,1:3,:3,2:5]))

        eri1 = ao2mo.incore.full(eri, mo[:,:0])
        self.assertTrue(eri1.size == 0)

    def test_incore_eri_s4(self):
        numpy.random.seed(1)
        norb = 4

        # A 4-index eri with 4-fold symmetry
        h2_s1 = numpy.random.random((norb, norb, norb, norb))
        h2_s1 = h2_s1 + h2_s1.transpose(1,0,2,3)
        h2_s1 = h2_s1 + h2_s1.transpose(0,1,3,2)

        # pack the eri to 2-index
        h2_s4 = ao2mo.restore(4, h2_s1, norb)

        mos = numpy.random.random((4,norb,norb-1))
        eri_mo_from_s4 = ao2mo.general(h2_s4, mos)
        eri_ref = trans(h2_s4, mos).reshape(eri_mo_from_s4.shape)

        self.assertAlmostEqual(abs(eri_mo_from_s4 - eri_ref).max(), 0, 12)


if __name__ == '__main__':
    print('Full Tests for ao2mo.incore')
    unittest.main()
