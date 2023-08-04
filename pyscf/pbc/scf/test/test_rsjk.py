#!/usr/bin/env python
# Copyright 2020-2021 The PySCF Developers. All Rights Reserved.
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
import numpy as np
import scipy.linalg
from pyscf.pbc.gto import Cell
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.scf import rsjk

def setUpModule():
    global cell, cell1
    cell = Cell().build(
         a = np.eye(3)*1.8,
         atom = '''He     0.      0.      0.
                   He     0.4917  0.4917  0.4917''',
         basis = {'He': [[0, [2.5, 1]]]})

    cell1 = Cell().build(
          a = np.eye(3)*2.6,
          atom = '''He     0.4917  0.4917  0.4917''',
          basis = {'He': [[0, [4.8, 1, -.1],
                              [1.1, .3, .5],
                              [0.15, .2, .8]],
                          [1, [0.8, 1]],]})


def tearDownModule():
    global cell, cell1
    del cell, cell1

class KnowValues(unittest.TestCase):
    def test_get_jk(self):
        kpts = cell.make_kpts([3,1,1])
        np.random.seed(1)
        dm = (np.random.rand(len(kpts), cell.nao, cell.nao) +
              np.random.rand(len(kpts), cell.nao, cell.nao) * 1j)
        dm = dm + dm.transpose(0,2,1).conj()
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        phase = k2gamma.get_phase(cell, kpts, kmesh)[1]
        dm = np.einsum('Rk,kuv,Sk->RSuv', phase.conj().T, dm, phase.T)
        dm = np.einsum('Rk,RSuv,Sk->kuv', phase, dm.real, phase.conj())

        mf = cell.KRHF(kpts=kpts)
        jref, kref = mf.get_jk(cell, dm, kpts=kpts)
        ej = np.einsum('kij,kji->', jref, dm)
        ek = np.einsum('kij,kji->', kref, dm) * .5

        jk_builder = rsjk.RangeSeparatedJKBuilder(cell, kpts)
        jk_builder.omega = 0.5
        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        vj, vk = mf.jk_method('RS').get_jk(cell, dm)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        mf = cell.KUHF(kpts=kpts)
        jref, kref = mf.get_jk(cell, np.array([dm, dm]))
        vj, vk = mf.jk_method('RS').get_jk(cell, np.array([dm, dm]))
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        mf = cell.KROHF(kpts=kpts)
        jref, kref = mf.get_jk(cell, dm)
        vj, vk = mf.jk_method('RS').get_jk(cell, dm)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        mf = cell.RHF(kpt=kpts[0])
        jref, kref = mf.get_jk(cell, dm[0])
        vj, vk = mf.jk_method('RS').get_jk(cell, dm[0])
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        mf = cell.UHF(kpt=kpts[0])
        jref, kref = mf.get_jk(cell, dm[[0,0]])
        vj, vk = mf.jk_method('RS').get_jk(cell, dm[[0,0]])
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        mf = cell.ROHF(kpt=kpts[0])
        jref, kref = mf.get_jk(cell, dm[0])
        vj, vk = mf.jk_method('RS').get_jk(cell, dm[0])
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

    def test_get_jk_high_cost(self):
        kpts = cell1.make_kpts([3,1,1])
        np.random.seed(1)
        dm = (np.random.rand(len(kpts), cell1.nao, cell1.nao) +
              np.random.rand(len(kpts), cell1.nao, cell1.nao) * 1j)
        dm = dm + dm.transpose(0,2,1).conj()
        kmesh = k2gamma.kpts_to_kmesh(cell1, kpts)
        phase = k2gamma.get_phase(cell1, kpts, kmesh)[1]
        dm = np.einsum('Rk,kuv,Sk->RSuv', phase.conj().T, dm, phase.T)
        dm = np.einsum('Rk,RSuv,Sk->kuv', phase, dm.real, phase.conj())

        mf = cell1.KRHF(kpts=kpts)
        jref, kref = mf.get_jk(cell1, dm, kpts=kpts)
        ej = np.einsum('kij,kji->', jref, dm)
        ek = np.einsum('kij,kji->', kref, dm) * .5

        jk_builder = rsjk.RangeSeparatedJKBuilder(cell1, kpts)
        jk_builder.omega = 0.5
        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)

        jk_builder.max_memory = 0
        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)

if __name__ == '__main__':
    print("Full Tests for rsjk")
    unittest.main()
