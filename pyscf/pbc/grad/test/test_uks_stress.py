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
from pyscf.pbc.dft.numint import NumInt
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.grad import uks_stress, uks
from pyscf.pbc.grad.uks_stress import _finite_diff_cells

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
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        xc = 'lda,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc))
        dat = uks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 5e-9

    def test_get_vxc_gga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        dm *= .5
        xc = 'pbe,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc))
        dat = uks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-8

    def test_get_vxc_mgga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        xc = 'm06,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc))
        dat = uks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 5e-9

    def test_get_j(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        dm *= .5
        xc = 'lda,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc))
        dat = uks_stress.get_vxc(mf_grad, cell, dm, with_j=True)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            vj1 = FFTDF(cell1).get_jk(dm.sum(axis=0), with_k=False)[0]
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            vj2 = FFTDF(cell2).get_jk(dm.sum(axis=0), with_k=False)[0]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('sij,ji->', dm, (vj1-vj2)) * .5
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
        mf = cell.UKS(xc=xc).run()
        mf_grad = uks.Gradients(mf)
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
        mf = cell.UKS(xc=xc).run()
        mf_grad = uks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
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
        mf = cell.UKS(xc=xc).run()
        mf_grad = uks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

if __name__ == "__main__":
    print("Full Tests for UKS Stress tensor")
    unittest.main()
