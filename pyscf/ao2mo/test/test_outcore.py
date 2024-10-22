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
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo

def setUpModule():
    global mol, mo, nao
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#'out_h2o'
    mol.atom.extend([
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])

    mol.basis = 'cc-pvdz'
    mol.build()
    nao = mol.nao_nr()
    numpy.random.seed(15)
    mo = numpy.random.random((nao,nao))
    mo = mo.copy(order='F')

def tearDownModule():
    global mol, mo
    del mol, mo

class KnownValues(unittest.TestCase):
    def test_nroutcore_grad(self):
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        erifile = ftmp.name
        eri_ao = mol.intor('int2e_ip1', aosym='s1').reshape(3,nao,nao,nao,nao)
        eriref = numpy.einsum('npjkl,pi->nijkl', eri_ao, mo)
        eriref = numpy.einsum('nipkl,pj->nijkl', eriref, mo)
        eriref = numpy.einsum('nijpl,pk->nijkl', eriref, mo)
        eriref = numpy.einsum('nijkp,pl->nijkl', eriref, mo)

        ao2mo.outcore.full(mol, mo, erifile, dataname='eri_mo',
                           intor='int2e_ip1_sph', aosym='s2kl', comp=3,
                           max_memory=10, ioblk_size=5, compact=False)
        feri = h5py.File(erifile,'r')
        eri1 = numpy.array(feri['eri_mo']).reshape(3,nao,nao,nao,nao)
        feri.close()
        self.assertTrue(numpy.allclose(eri1, eriref))

    def test_nroutcore_eri(self):
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        erifile = ftmp.name
        eri_ao = ao2mo.restore(1, mol.intor('int2e', aosym='s2kl'), nao)
        eriref = numpy.einsum('pjkl,pi->ijkl', eri_ao, mo)
        eriref = numpy.einsum('ipkl,pj->ijkl', eriref, mo)
        eriref = numpy.einsum('ijpl,pk->ijkl', eriref, mo)
        eriref = numpy.einsum('ijkp,pl->ijkl', eriref, mo)

        mos = (mo[:,:4], mo[:,:3], mo[:,:3], mo[:,:2])
        ao2mo.outcore.general(mol, mos, erifile, dataname='eri_mo',
                              intor='int2e', aosym=1)
        with ao2mo.load(erifile) as eri1:
            eri1 = numpy.asarray(eri1).reshape(4,3,3,2)
        self.assertTrue(numpy.allclose(eri1, eriref[:4,:3,:3,:2]))

        ao2mo.outcore.full(mol, mo, erifile, dataname='eri_mo',
                              intor='int2e_sph', aosym='s2ij', comp=1)
        with ao2mo.load(erifile, 'eri_mo') as eri:
            eri1 = ao2mo.restore(1, numpy.array(eri), nao)
        eri1 = eri1.reshape(nao,nao,nao,nao)
        self.assertTrue(numpy.allclose(eri1, eriref))

        mos = (mo[:,:3], mo[:,:3], mo[:,:3], mo[:,:2])
        ao2mo.outcore.general(mol, mos, erifile, dataname='eri_mo',
                              intor='int2e_sph', aosym='s4', comp=1,
                              compact=False)
        with ao2mo.load(erifile, 'eri_mo') as eri1:
            eri1 = numpy.asarray(eri1).reshape(3,3,3,2)
        self.assertTrue(numpy.allclose(eri1, eriref[:3,:3,:3,:2]))

        ao2mo.outcore.full(mol, mo[:,:0], erifile,
                           intor='int2e', aosym='1', comp=1)
        with ao2mo.load(erifile, 'eri_mo') as eri:
            self.assertTrue(eri.size == 0)

    def test_group_segs(self):
        numpy.random.seed(1)
        segs = numpy.asarray(numpy.random.random(40)*50, dtype=int)
        ref = [(0, 7, 91), (7, 11, 82), (11, 15, 88), (15, 20, 96), (20, 22, 88),
               (22, 25, 92), (25, 30, 100), (30, 34, 98), (34, 37, 83), (37, 40, 78)]
        out = ao2mo.outcore.balance_segs(segs, 100)
        self.assertTrue(ref == out)

    def test_ao2mo_with_mol_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        nao = pmol.nao_nr()
        numpy.random.seed(1)
        mo = numpy.random.random((nao,4))
        eri = ao2mo.kernel(pmol, mo)
        self.assertAlmostEqual(lib.fp(eri), -977.99841341828437, 9)

        eri = ao2mo.kernel(mol, mo, intor='int2e_cart')
        self.assertAlmostEqual(lib.fp(eri), -977.99841341828437, 9)

if __name__ == '__main__':
    print('Full Tests for ao2mo.outcore')
    unittest.main()
