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
import numpy
from pyscf import ao2mo
from pyscf.pbc.df import aft
import pyscf.pbc.gto as pgto
from pyscf.pbc.lib import kpts_helper

def setUpModule():
    global cell, nao
    L = 5.
    n = 3
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    nao = cell.nao_nr()

def tearDownModule():
    global cell
    del cell


class KnownValues(unittest.TestCase):
    def test_eri1111(self):
        kpts = numpy.random.random((4,3)) * .25
        kpts[3] = -numpy.einsum('ij->j', kpts[:3])
        with_df = aft.AFTDF(cell)
        with_df.kpts = kpts
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri = with_df.get_eri(kpts).reshape((nao,)*4)
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, kpts)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).max(), 0, 9)

    def test_eri0110(self):
        kpts = numpy.random.random((4,3)) * .25
        kpts[3] = kpts[0]
        kpts[2] = kpts[1]
        with_df = aft.AFTDF(cell)
        with_df.kpts = kpts
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri = with_df.get_eri(kpts).reshape((nao,)*4)
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, kpts)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).max(), 0, 9)

    def test_eri0000(self):
        with_df = aft.AFTDF(cell)
        with_df.kpts = numpy.zeros((4,3))
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri = ao2mo.restore(1, with_df.get_eri(with_df.kpts), nao)
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, with_df.kpts)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).max(), 0, 9)

        mo = mo.real
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, with_df.kpts, compact=False)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).max(), 0, 9)

    def test_ao2mo_7d(self):
        L = 3.
        n = 6
        cell = pgto.Cell()
        cell.a = numpy.diag([L,L,L])
        cell.mesh = [n,n,n]
        cell.atom = '''He    2.    2.2      2.
                       He    1.2   1.       1.'''
        cell.basis = {'He': [[0, (1.2, 1)], [1, (0.6, 1)]]}
        cell.verbose = 0
        cell.precision = 1e-12
        cell.build(0,0)

        kpts = cell.make_kpts([1,3,1])
        nkpts = len(kpts)
        nao = cell.nao_nr()
        numpy.random.seed(1)
        mo =(numpy.random.random((nkpts,nao,nao)) +
             numpy.random.random((nkpts,nao,nao))*1j)

        with_df = aft.AFTDF(cell, kpts)
        out = with_df.ao2mo_7d(mo, kpts)
        ref = numpy.empty_like(out)

        kconserv = kpts_helper.get_kconserv(cell, kpts)
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kj, kk]
            tmp = with_df.ao2mo((mo[ki], mo[kj], mo[kk], mo[kl]), kpts[[ki,kj,kk,kl]])
            ref[ki,kj,kk] = tmp.reshape([nao]*4)

        self.assertAlmostEqual(abs(out-ref).max(), 0, 12)

if __name__ == '__main__':
    print("Full Tests for aft ao2mo")
    unittest.main()
