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
from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.pbc.df import FFTDF
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.scf import rsjk

class KnownValues(unittest.TestCase):
    def test_get_jk(self):
        cell = Cell().build(
             a = np.eye(3)*1.8,
             atom = '''He     0.      0.      0.
                       He     0.4917  0.4917  0.4917''',
             basis = {'He': [[0, [2.5, 1]]]},
             precision = 1e-9,
        )

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
        jk_builder.allow_drv_nodddd = False
        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        vj, vk = mf.jk_method('RS').get_jk(cell, dm)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        mf = cell.KUHF(kpts=kpts)
        jref, kref = mf.get_jk(cell, np.array([dm, dm]))
        vj, vk = mf.jk_method('RS').get_jk(cell, np.array([dm, dm]))
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        mf = cell.KROHF(kpts=kpts)
        jref, kref = mf.get_jk(cell, dm)
        vj, vk = mf.jk_method('RS').get_jk(cell, dm)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        mf = cell.RHF(kpt=kpts[0])
        jref, kref = mf.get_jk(cell, dm[0])
        vj, vk = mf.jk_method('RS').get_jk(cell, dm[0])
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        mf = cell.UHF(kpt=kpts[0])
        jref, kref = mf.get_jk(cell, dm[[0,0]])
        vj, vk = mf.jk_method('RS').get_jk(cell, dm[[0,0]])
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

        mf = cell.ROHF(kpt=kpts[0])
        jref, kref = mf.get_jk(cell, dm[0])
        vj, vk = mf.jk_method('RS').get_jk(cell, dm[0])
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 8)

    def test_get_jk_fft_dd_block(self):
        cell1 = Cell().build(
            a = np.eye(3)*2.6,
            atom = '''He     0.4917  0.4917  0.4917''',
            basis = {'He': [[0, [4.8, 1, -.1],
                             [1.1, .3, .5],
                             [0.1, .2, .8]],
                            [1, [0.8, 1]],]})
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
        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 6)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)

        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 6)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 6)

        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_k=False)
        self.assertAlmostEqual(abs(vj - jref).max(), 0, 6)

        jk_builder.max_memory = 0
        vj, vk = jk_builder.get_jk(dm, hermi=0, kpts=kpts, exxdiv=mf.exxdiv, with_j=False)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 6)

    def test_get_jk_sr_without_dddd(self):
        cell1 = Cell().build(
            a = np.eye(3)*2.6,
            atom = '''He     0.4917  0.4917  0.4917''',
            basis = {'He': [[0, [4.8, 1, -.1],
                             [1.1, .3, .5],
                             [0.1, .2, .8]],]})
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
        jk_builder.exclude_dd_block = True
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

    def test_2d(self):
        a = np.eye(3) * 2.5
        a[2,2] = 7.
        cell1 = Cell().build(
            dimension = 2,
            a = a,
            atom = '''He 0.5 0.5 0.2''',
            basis = {'He': [[0, [4.0, 1, -.1],
                             [1.1, .3, .5]],]})
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

    def test_update_vk_high_cost(self):
        L = 5.
        cell = Cell()
        np.random.seed(1)
        cell.a = np.diag([L,L,L]) + np.random.rand(3,3)
        cell.mesh = np.array([27]*3)
        cell.atom = '''H    1.8   2.       .4
                       Li    1.    1.       1.'''
        cell.basis = {'H': [[0, (1.0, 1)],
                            [1, (0.8, 1)]],
                      'Li': [[0, (1.0, 1)],
                             [0, (0.6, 1)],
                             [1, (0.8, 1)]]}
        cell.build()
        kmesh = [6,1,1]
        kpts = cell.make_kpts(kmesh)
        mf = cell.KRHF(kpts=kpts)
        mf.run()
        mf.mo_coeff = np.array(mf.mo_coeff)

        dm = mf.make_rdm1()
        refj, refk = FFTDF(cell, kpts=kpts).get_jk(dm, kpts=kpts)
        self.assertAlmostEqual(lib.fp(refk), -7.614964681516531, 8)

        mf.jk_method('RS')
        vj, vk = mf.rsjk.get_jk(dm, kpts=kpts, exxdiv=None)
        self.assertAlmostEqual(abs(vj-refj).max(), 0, 7)
        self.assertAlmostEqual(abs(vk-refk).max(), 0, 7)

        mf.rsjk.reset()
        mf.rsjk.exclude_dd_block = True
        mf.rsjk.allow_drv_nodddd = False
        vj, vk = mf.rsjk.get_jk(dm, kpts=kpts, exxdiv=None)
        self.assertAlmostEqual(abs(vj-refj).max(), 0, 7)
        self.assertAlmostEqual(abs(vk-refk).max(), 0, 7)

        mf.rsjk.reset()
        mf.rsjk.exclude_dd_block = False
        mf.rsjk.allow_drv_nodddd = True
        vj, vk = mf.rsjk.get_jk(dm, kpts=kpts, exxdiv=None)
        self.assertAlmostEqual(abs(vj-refj).max(), 0, 7)
        self.assertAlmostEqual(abs(vk-refk).max(), 0, 7)

        mf.rsjk.reset()
        mf.rsjk.exclude_dd_block = False
        mf.rsjk.allow_drv_nodddd = False
        vj, vk = mf.rsjk.get_jk(dm, kpts=kpts, exxdiv=None)
        self.assertAlmostEqual(abs(vj-refj).max(), 0, 7)
        self.assertAlmostEqual(abs(vk-refk).max(), 0, 7)

if __name__ == '__main__':
    print("Full Tests for rsjk")
    unittest.main()
