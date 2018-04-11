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
naopair = nao*(nao+1)/2
numpy.random.seed(15)
mo = numpy.random.random((nao,nao))
mo = mo.copy(order='F')

c_atm = numpy.array(mol._atm, dtype=numpy.int32)
c_bas = numpy.array(mol._bas, dtype=numpy.int32)
c_env = numpy.array(mol._env)
natm = ctypes.c_int(c_atm.shape[0])
nbas = ctypes.c_int(c_bas.shape[0])

class KnownValues(unittest.TestCase):
    def test_nroutcore_grad(self):
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        erifile = ftmp.name
        eri_ao = numpy.empty((3,nao,nao,nao,nao))
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                kp = 0
                for k in range(mol.nbas):
                    lp = 0
                    for l in range(mol.nbas):
                        buf = mol.intor_by_shell('int2e_ip1_sph', (i,j,k,l), comp=3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri_ao[:,ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                        lp += dl
                    kp += dk
                jp += dj
            ip += di
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
        eri_ao = numpy.empty((nao,nao,nao,nao))
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                kp = 0
                for k in range(mol.nbas):
                    lp = 0
                    for l in range(mol.nbas):
                        buf = mol.intor_by_shell('int2e_sph', (i,j,k,l))
                        di,dj,dk,dl = buf.shape
                        eri_ao[ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                        lp += dl
                    kp += dk
                jp += dj
            ip += di
        eriref = numpy.einsum('pjkl,pi->ijkl', eri_ao, mo)
        eriref = numpy.einsum('ipkl,pj->ijkl', eriref, mo)
        eriref = numpy.einsum('ijpl,pk->ijkl', eriref, mo)
        eriref = numpy.einsum('ijkp,pl->ijkl', eriref, mo)

        ao2mo.outcore.full(mol, mo, erifile, dataname='eri_mo',
                           intor='int2e_sph', aosym='s1', comp=1,
                           max_memory=10, ioblk_size=5)
        feri = h5py.File(erifile)
        eri1 = numpy.array(feri['eri_mo']).reshape(nao,nao,nao,nao)
        feri.close()
        self.assertTrue(numpy.allclose(eri1, eriref))

        ao2mo.outcore.full(mol, mo, erifile, dataname='eri_mo',
                           intor='int2e_sph', aosym='s2ij', comp=1,
                           max_memory=10, ioblk_size=5)
        feri = h5py.File(erifile)
        eri1 = s2ij_s1(1, numpy.array(feri['eri_mo']), nao)
        feri.close()
        eri1 = eri1.reshape(nao,nao,nao,nao)
        self.assertTrue(numpy.allclose(eri1, eriref))

        ao2mo.outcore.full(mol, mo, erifile, dataname='eri_mo',
                           intor='int2e_sph', aosym='s2kl', comp=1,
                           max_memory=10, ioblk_size=5)
        feri = h5py.File(erifile)
        eri1 = s2kl_s1(1, numpy.array(feri['eri_mo']), nao)
        feri.close()
        eri1 = eri1.reshape(nao,nao,nao,nao)
        self.assertTrue(numpy.allclose(eri1, eriref))

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
        self.assertAlmostEqual(lib.finger(eri), -977.99841341828437, 9)

def s2ij_s1(symmetry, eri, norb):
    idx = numpy.tril_indices(norb)
    eri1 = numpy.empty((norb,norb,norb,norb))
    eri1[idx] = eri.reshape(-1,norb,norb)
    eri1[idx[1],idx[0]] = eri.reshape(-1,norb,norb)
    return eri1

def s2kl_s1(symmetry, eri, norb):
    idx = numpy.tril_indices(norb)
    eri1 = numpy.empty((norb,norb,norb,norb))
    eri1[:,:,idx[0],idx[1]] = eri.reshape(norb,norb,-1)
    eri1[:,:,idx[1],idx[0]] = eri.reshape(norb,norb,-1)
    return eri1

if __name__ == '__main__':
    print('Full Tests for outcore')
    unittest.main()

