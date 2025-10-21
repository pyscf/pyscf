#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
from pyscf.gto import ATOM_OF, intor_cross
from pyscf.pbc import dft, gto, grad
from pyscf.pbc.tools import pbc
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.numint import KNumInt
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.dft import kukspu
from pyscf.pbc.grad import kuks_stress, kuks
from pyscf.pbc.grad.kuks_stress import _finite_diff_cells

def setUpModule():
    global cell
    a = np.eye(3) * 5
    np.random.seed(5)
    a += np.random.rand(3, 3) - .5
    cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                 basis=[[0, [.5, 1]], [1, [.5, 1]]], a=a, unit='Bohr')

class KnownValues(unittest.TestCase):
    def test_get_vxc_lda(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(2,np.prod(kmesh), nao, nao) - (.5+.1j)
        dm = np.einsum('skpi,skqi->skpq', dm, dm.conj())
        xc = 'lda,'
        mf_grad = kuks.Gradients(cell.KUKS(xc=xc, kpts=cell.make_kpts(kmesh)))
        dat = kuks_stress.get_vxc(mf_grad, cell, dm, kpts=cell.make_kpts(kmesh))
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 2e-9

    def test_get_vxc_gga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(2,np.prod(kmesh), nao, nao) - (.5+.1j)
        dm = np.einsum('skpi,skqi->skpq', dm, dm.conj())
        xc = 'pbe,'
        mf_grad = kuks.Gradients(cell.KUKS(xc=xc, kpts=cell.make_kpts(kmesh)))
        dat = kuks_stress.get_vxc(mf_grad, cell, dm, kpts=cell.make_kpts(kmesh))
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-8

    def test_get_vxc_mgga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(2,np.prod(kmesh), nao, nao) - (.5+.1j)
        dm = np.einsum('skpi,skqi->skpq', dm, dm.conj())
        xc = 'm06,'
        mf_grad = kuks.Gradients(cell.KUKS(xc=xc, kpts=cell.make_kpts(kmesh)))
        dat = kuks_stress.get_vxc(mf_grad, cell, dm, kpts=cell.make_kpts(kmesh))
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 5e-9

    def test_get_j(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 3]
        nao = cell.nao
        dm = np.random.rand(2,np.prod(kmesh), nao, nao) - (.5+.1j)
        dm *= .5
        dm = np.einsum('skpi,skqi->skpq', dm, dm.conj())
        xc = 'lda,'
        kpts = cell.make_kpts(kmesh)
        mf_grad = kuks.Gradients(cell.KUKS(xc=xc, kpts=kpts))
        dat = kuks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_j=True)
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            vj1 = FFTDF(cell1).get_jk(dm.sum(axis=0), kpts=cell1.make_kpts(kmesh), with_k=False)[0]
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            vj2 = FFTDF(cell2).get_jk(dm.sum(axis=0), kpts=cell2.make_kpts(kmesh), with_k=False)[0]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            de = np.einsum('skij,kji->', dm, (vj1-vj2)) / len(kpts) * .5
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 1e-8

    def test_lda_vs_finite_difference(self):
        a = np.eye(3) * 3
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='H 1 1 1; H 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     a=a, unit='Bohr', verbose=0)
        xc = 'svwn'
        kmesh = [3, 1, 1]
        mf = cell.KUKS(xc=xc, kpts=cell.make_kpts(kmesh)).run()
        mf_grad = kuks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

    def test_gga_vs_finite_difference_high_cost(self):
        a = np.eye(3) * 3.5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='B 1 1 1; C 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     spin=1,
                     pseudo='gth-pade', a=a, unit='Bohr', verbose=0)
        xc = 'pbe'
        kmesh = [3, 1, 1]
        mf = cell.KUKS(xc=xc, kpts=cell.make_kpts(kmesh)).run()
        mf_grad = kuks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

    def test_mgga_vs_finite_difference_high_cost(self):
        a = np.eye(3) * 3.5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='H 1 1 1; H 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     a=a, unit='Bohr', verbose=0)
        xc = 'rscan'
        kmesh = [3, 1, 1]
        mf = cell.KUKS(xc=xc, kpts=cell.make_kpts(kmesh)).run()
        mf_grad = kuks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

    def test_hubbard_U(self):
        cell = gto.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; O 0.5,  0.8,  1.1',
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
            pseudo = 'gth-pbe')
        kmesh = [3,1,1]
        kpts = cell.make_kpts(kmesh)
        minao = 'gth-szv'

        U_idx = ['C 2p']
        U_val = [5]
        mf = kukspu.KUKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao)
        mf.__dict__.update(cell.KUKS(kpts=kpts).run(max_cycle=1).__dict__)
        sigma = kuks_stress._hubbard_U_deriv1(mf)

        for (i, j) in [(1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            mf.reset(cell1)
            e1 = mf.get_veff().E_U.real
            mf.reset(cell2)
            e2 = mf.get_veff().E_U.real
            assert abs(sigma[i,j] - (e1 - e2) / 2e-4) < 1e-8

    def test_kukspu_finite_diff_high_cost(self):
        cell = gto.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; O 0.5,  0.8,  1.1',
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
            pseudo = 'gth-pbe')
        kmesh = [3,1,1]
        kpts = cell.make_kpts(kmesh)
        minao = 'gth-szv'

        U_idx = ['C 2p']
        U_val = [5]
        mf = kukspu.KUKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao).run()
        sigma = mf.Gradients().get_stress()
        mf_scanner = mf.as_scanner()

        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-3)
        e1 = mf_scanner(cell1)
        e2 = mf_scanner(cell2)
        assert abs(sigma[0,0] - (e1 - e2)/2e-3/cell.vol) < 1e-6

if __name__ == "__main__":
    print("Full Tests for KUKS Stress tensor")
    unittest.main()
