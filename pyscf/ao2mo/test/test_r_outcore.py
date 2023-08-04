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
import tempfile
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo

def setUpModule():
    global mol, eri0
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.build()
    eri0 = mol.intor('int2e_spinor')

def tearDownModule():
    global mol, eri0
    mol.stdout.close()
    del mol, eri0

def trans(eri, mos):
    eriref = lib.einsum('pjkl,pi->ijkl', eri   , mos[0].conj())
    eriref = lib.einsum('ipkl,pj->ijkl', eriref, mos[1])
    eriref = lib.einsum('ijpl,pk->ijkl', eriref, mos[2].conj())
    eriref = lib.einsum('ijkp,pl->ijkl', eriref, mos[3])
    return eriref

class KnownValues(unittest.TestCase):
    def test_r_outcore_eri(self):
        n2c = mol.nao_2c()
        numpy.random.seed(1)
        mo = numpy.random.random((n2c,n2c)) + numpy.random.random((n2c,n2c))*1j
        eriref = trans(eri0, [mo]*4)
        ftmp = tempfile.NamedTemporaryFile()

        ao2mo.kernel(mol, mo, erifile=ftmp.name, intor='int2e_spinor', max_memory=10, ioblk_size=5)
        with ao2mo.load(ftmp) as eri1:
            self.assertAlmostEqual(lib.fp(eri1), -550.72966498073129-1149.3561026721848j, 8)
            self.assertAlmostEqual(abs(eri1-eriref.reshape(n2c**2,n2c**2)).max(), 0, 9)

        eri1 = ao2mo.kernel(mol, (mo[:,:2], mo[:,:4], mo[:,:2], mo[:,:4]),
                            erifile=ftmp.name, intor='int2e_spinor')
        with ao2mo.load(ftmp) as eri1:
            self.assertAlmostEqual(abs(eri1-eriref[:2,:4,:2,:4].reshape(8,8)).max(), 0, 9)

        ftmp = lib.H5TmpFile()
        ao2mo.kernel(mol, (mo[:,:2], mo[:,:4], mo[:,:2], mo[:,:4]),
                     erifile=ftmp, intor='int2e_spinor', aosym='s1')
        with ao2mo.load(ftmp) as eri1:
            self.assertAlmostEqual(abs(eri1-eriref[:2,:4,:2,:4].reshape(8,8)).max(), 0, 9)

        eri1 = ao2mo.kernel(mol, (mo[:,:2], mo[:,:4], mo[:,:4], mo[:,:2]),
                            intor='int2e_spinor', aosym='s2ij')
        self.assertAlmostEqual(abs(eri1-eriref[:2,:4,:4,:2].reshape(8,8)).max(), 0, 9)

        eri1 = ao2mo.kernel(mol, (mo[:,:2], mo[:,:4], mo[:,:2], mo[:,:4]),
                            intor='int2e_spinor', aosym='s2kl')
        self.assertAlmostEqual(abs(eri1-eriref[:2,:4,:2,:4].reshape(8,8)).max(), 0, 9)

        eri1 = ao2mo.kernel(mol, mo[:,:0], intor='int2e_spinor')
        self.assertTrue(eri1.size == 0)

    def test_r_outcore_eri_grad(self):
        n2c = mol.nao_2c()
        numpy.random.seed(1)
        mo = numpy.random.random((n2c,4)) + numpy.random.random((n2c,4))*1j
        eri1 = ao2mo.kernel(mol, mo, intor='int2e_ip1_spinor')
        self.assertAlmostEqual(lib.fp(eri1), -696.47505768925771-265.10054236197817j, 8)

    def test_ao2mo_r_e2(self):
        n2c = mol.nao_2c()
        numpy.random.seed(1)
        mo = numpy.random.random((n2c,n2c)) + numpy.random.random((n2c,n2c))*1j
        tao = numpy.asarray(mol.tmap(), dtype=numpy.int32)
        buf = ao2mo._ao2mo.r_e1('int2e_spinor', mo, (0,4,0,3), (0, 2, 8),
                                mol._atm, mol._bas, mol._env, tao, 's1')
        buf = buf.reshape(8,12).T
        ref = lib.einsum('pqkl,pi,qj->ijkl', eri0, mo[:,:4].conj(), mo[:,:3])
        self.assertAlmostEqual(lib.fp(buf), 0.30769732102451997-0.58664393190628461j, 8)
        self.assertAlmostEqual(abs(buf[:,:4]-ref[:,:,:2,:2].reshape(12,4)).max(), 0, 9)
        self.assertAlmostEqual(abs(buf[:,4:]-ref[:,:,:2,2:4].reshape(12,4)).max(), 0, 9)

        buf = ao2mo._ao2mo.r_e2(eri0.reshape(n2c**2,n2c,n2c), mo, (0,2,0,4), tao, None, 's1')
        ref = lib.einsum('xpq,pk,ql->xkl', eri0.reshape(n2c**2,n2c,n2c),
                         mo[:,:2].conj(), mo[:,:4])
        self.assertAlmostEqual(lib.fp(buf), 14.183520455200011+10.179224253811057j, 8)
        self.assertAlmostEqual(abs(buf.reshape(n2c**2,2,4)-ref).max(), 0, 9)

        buf = ao2mo._ao2mo.r_e2(eri0.reshape(n2c**2,n2c,n2c), mo, (0,0,4,4), tao, None, 's1')
        self.assertEqual(buf.size, 0)


if __name__ == '__main__':
    print('Full Tests for ao2mo.r_outcore')
    unittest.main()
#
#if __name__ == '__main__':
#    from pyscf import scf
#    from pyscf import gto
#    from pyscf.ao2mo import addons
#    mol = gto.M(
#        verbose = 0,
#        atom = [
#            ["O" , (0. , 0.     , 0.)],
#            [1   , (0. , -0.757 , 0.587)],
#            [1   , (0. , 0.757  , 0.587)]],
#        basis = 'ccpvdz')
#
#    mf = scf.RHF(mol)
#    mf.scf()
#
#    eri0 = full(mf._eri, mf.mo_coeff)
#    mos = (mf.mo_coeff,)*4
#    print(numpy.allclose(eri0, full(mol, mf.mo_coeff)))
#    print(numpy.allclose(eri0, general(mf._eri, mos)))
#    print(numpy.allclose(eri0, general(mol, mos)))
#    with load(full(mol, mf.mo_coeff, 'h2oeri.h5', dataname='dat1'), 'dat1') as eri1:
#        print(numpy.allclose(eri0, eri1))
#    with load(general(mol, mos, 'h2oeri.h5', dataname='dat1'), 'dat1') as eri1:
#        print(numpy.allclose(eri0, eri1))
#
