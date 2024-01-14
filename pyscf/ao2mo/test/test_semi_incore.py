#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf.ao2mo import semi_incore

def setUpModule():
    global mol, nao, eri
    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '6311g'
    mol.build()
    nao = mol.nao_nr()
    eri = mol.intor('int2e_sph', aosym='s8')

def tearDownModule():
    global mol, eri
    del mol, eri

class KnowValues(unittest.TestCase):
    def test_general(self):
        numpy.random.seed(15)
        nmo = 12
        mo = numpy.random.random((nao,nmo))
        eriref = ao2mo.incore.full(eri, mo)

        tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        io_size = nao**2*4e-5

        semi_incore.general(eri, [mo]*4, tmpfile.name, ioblk_size=io_size)
        with ao2mo.load(tmpfile) as eri_mo:
            self.assertAlmostEqual(abs(eriref - eri_mo[:]).max(), 0, 9)

        semi_incore.general(eri, [mo]*4, tmpfile.name, ioblk_size=io_size,
                            compact=False)
        with ao2mo.load(tmpfile) as eri_mo:
            eriref = ao2mo.restore(1, eriref, nmo).reshape(nmo**2,nmo**2)
            self.assertAlmostEqual(abs(eriref - eri_mo[:]).max(), 0, 9)

    def test_general_complex(self):
        numpy.random.seed(15)
        nmo = 12
        mo = numpy.random.random((nao,nmo)) + numpy.random.random((nao,nmo))*1j
        eriref = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', ao2mo.restore(1, eri, nao),
                            mo.conj(), mo, mo.conj(), mo)
        eriref = eriref.reshape(12**2,12**2)

        tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        io_size = nao**2*4e-5

        semi_incore.general(eri, [mo]*4, tmpfile.name, ioblk_size=io_size)
        with ao2mo.load(tmpfile) as eri_mo:
            self.assertAlmostEqual(abs(eriref - eri_mo[:]).max(), 0, 9)

        io_size = nao**2*4e-5
        eri_4fold = ao2mo.restore(4, eri, nao)
        semi_incore.general(eri_4fold, [mo]*4, tmpfile.name, ioblk_size=io_size)
        with ao2mo.load(tmpfile) as eri_mo:
            self.assertAlmostEqual(abs(eriref - eri_mo[:]).max(), 0, 9)


if __name__ == '__main__':
    print('Full Tests for ao2mo.semi_incore')
    unittest.main()
