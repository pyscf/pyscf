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
from pyscf import lib
from pyscf.gto import intor_cross
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import pbc
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.numint import KNumInt
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.dft import krkspu
from pyscf.pbc.grad import krks_stress, krks
from pyscf.pbc.grad.krks_stress import _finite_diff_cells

def setUpModule():
    global cell
    a = np.eye(3) * 5
    np.random.seed(5)
    a += np.random.rand(3, 3) - .5
    cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                 basis=[[0, [.5, 1]], [1, [.5, 1]]], a=a, unit='Bohr')

class KnownValues(unittest.TestCase):
    def test_ovlp(self):
        kmesh = [3,1,1]
        sigma9 = krks_stress.get_ovlp(cell, cell.make_kpts(kmesh))
        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-5)
        s1 = np.asarray(cell1.pbc_intor('int1e_ovlp', kpts=cell1.make_kpts(kmesh)))
        s2 = np.asarray(cell2.pbc_intor('int1e_ovlp', kpts=cell2.make_kpts(kmesh)))
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[0] - ref).max() < 1e-9

        cell1, cell2 = _finite_diff_cells(cell, 0, 1, disp=1e-5)
        s1 = np.asarray(cell1.pbc_intor('int1e_ovlp', kpts=cell1.make_kpts(kmesh)))
        s2 = np.asarray(cell2.pbc_intor('int1e_ovlp', kpts=cell2.make_kpts(kmesh)))
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[1] - ref).max() < 1e-9

        aoslices = cell.aoslice_by_atom()
        ao_repeats = aoslices[:,3] - aoslices[:,2]
        bas_coords = np.repeat(cell.atom_coords(), ao_repeats, axis=0)
        ovlp10 = cell.pbc_intor('int1e_ipovlp', kpts=cell.make_kpts(kmesh))
        ovlp10 = np.einsum('kxij,iy->xykij', ovlp10, bas_coords)

        nao = cell.nao
        Ls = cell.get_lattice_Ls()
        scell = cell.copy()
        scell = pbc._build_supcell_(scell, cell, Ls)
        sc_ovlp01 = intor_cross('int1e_ipovlp', scell, cell)
        sc_ovlp01 = sc_ovlp01.transpose(0,2,1).reshape(3,nao,-1,nao)
        expLk = np.exp(1j*np.dot(Ls, cell.make_kpts(kmesh).T))
        ovlp01 = np.einsum('xinj,njy,nk->xykij', sc_ovlp01, bas_coords+Ls[:,None], expLk)
        dat = -(ovlp10 + ovlp01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

        sc_ovlp10 = intor_cross('int1e_ipovlp', cell, scell)
        sc_ovlp10 = sc_ovlp10.reshape(3,nao,-1,nao)
        ovlp01 = np.einsum('xinj,njy,nk->xykij', sc_ovlp10, bas_coords+Ls[:,None], expLk)
        dat = -(ovlp10 - ovlp01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

    def test_kin(self):
        kmesh = [3,1,1]
        sigma9 = krks_stress.get_kin(cell, cell.make_kpts(kmesh))
        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-5)
        s1 = np.asarray(cell1.pbc_intor('int1e_kin', kpts=cell1.make_kpts(kmesh)))
        s2 = np.asarray(cell2.pbc_intor('int1e_kin', kpts=cell2.make_kpts(kmesh)))
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[0] - ref).max() < 1e-9

        cell1, cell2 = _finite_diff_cells(cell, 0, 1, disp=1e-5)
        s1 = np.asarray(cell1.pbc_intor('int1e_kin', kpts=cell1.make_kpts(kmesh)))
        s2 = np.asarray(cell2.pbc_intor('int1e_kin', kpts=cell2.make_kpts(kmesh)))
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[1] - ref).max() < 1e-9

        aoslices = cell.aoslice_by_atom()
        ao_repeats = aoslices[:,3] - aoslices[:,2]
        bas_coords = np.repeat(cell.atom_coords(), ao_repeats, axis=0)
        kin10 = cell.pbc_intor('int1e_ipkin', kpts=cell.make_kpts(kmesh))
        kin10 = np.einsum('kxij,iy->xykij', kin10, bas_coords)

        nao = cell.nao
        Ls = cell.get_lattice_Ls()
        scell = cell.copy()
        scell = pbc._build_supcell_(scell, cell, Ls)
        sc_kin01 = intor_cross('int1e_ipkin', scell, cell)
        sc_kin01 = sc_kin01.transpose(0,2,1).reshape(3,nao,-1,nao)
        expLk = np.exp(1j*np.dot(Ls, cell.make_kpts(kmesh).T))
        kin01 = np.einsum('xinj,njy,nk->xykij', sc_kin01, bas_coords+Ls[:,None], expLk)
        dat = -(kin10 + kin01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

        sc_kin10 = intor_cross('int1e_ipkin', cell, scell)
        sc_kin10 = sc_kin10.reshape(3,nao,-1,nao)
        kin01 = np.einsum('xinj,njy,nk->xykij', sc_kin10, bas_coords+Ls[:,None], expLk)
        dat = -(kin10 - kin01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

    def test_eval_ao_kpts(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        coords = np.random.rand(10, 3)
        ao_value = krks_stress._eval_ao_strain_derivatives(cell, coords, kpts)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao_kpts(cell1, coords, cell1.make_kpts(kmesh))
            ao2 = dft.numint.eval_ao_kpts(cell2, coords, cell2.make_kpts(kmesh))
            assert abs(ao_value[0][i,j,0] - (ao1[0] - ao2[0]) / 2e-5).max() < 1e-9
            assert abs(ao_value[1][i,j,0] - (ao1[1] - ao2[1]) / 2e-5).max() < 1e-9
            assert abs(ao_value[2][i,j,0] - (ao1[2] - ao2[2]) / 2e-5).max() < 1e-9

    def test_eval_ao_deriv1_cart_kpts(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]]], a=a, unit='Bohr', cart=True)
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        coords = np.random.rand(10, 3)
        ao_value = krks_stress._eval_ao_strain_derivatives(cell, coords, kpts, deriv=1)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao_kpts(cell1, coords, cell1.make_kpts(kmesh), deriv=1)
            ao2 = dft.numint.eval_ao_kpts(cell2, coords, cell2.make_kpts(kmesh), deriv=1)
            assert abs(ao_value[0][i,j] - (ao1[0] - ao2[0]) / 2e-5).max() < 1e-9
            assert abs(ao_value[1][i,j] - (ao1[1] - ao2[1]) / 2e-5).max() < 1e-9
            assert abs(ao_value[2][i,j] - (ao1[2] - ao2[2]) / 2e-5).max() < 1e-9

    def test_eval_ao_deriv1_sph_kpts(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        coords = np.random.rand(10, 3)
        ao_value = krks_stress._eval_ao_strain_derivatives(cell, coords, kpts, deriv=1)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao_kpts(cell1, coords, cell1.make_kpts(kmesh), deriv=1)
            ao2 = dft.numint.eval_ao_kpts(cell2, coords, cell2.make_kpts(kmesh), deriv=1)
            assert abs(ao_value[0][i,j] - (ao1[0] - ao2[0]) / 2e-5).max() < 1e-9
            assert abs(ao_value[1][i,j] - (ao1[1] - ao2[1]) / 2e-5).max() < 1e-9
            assert abs(ao_value[2][i,j] - (ao1[2] - ao2[2]) / 2e-5).max() < 1e-9

    def test_get_vxc_lda(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - (.5+.1j)
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        kmesh = [3, 1, 1]
        mf_grad = krks.Gradients(cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)))
        dat = krks_stress.get_vxc(mf_grad, cell, dm, kpts=cell.make_kpts(kmesh))
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_vxc_gga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(len(kmesh), nao, nao) - (.5+.2j)
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'pbe,'
        mf_grad = krks.Gradients(cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)))
        dat = krks_stress.get_vxc(mf_grad, cell, dm, kpts=cell.make_kpts(kmesh))
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_vxc_mgga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - (.5+.1j)
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'm06,'
        mf_grad = krks.Gradients(cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)))
        dat = krks_stress.get_vxc(mf_grad, cell, dm, kpts=cell.make_kpts(kmesh))
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_j(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - (.5+.1j)
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        kpts = cell.make_kpts(kmesh)
        mf_grad = krks.Gradients(cell.KRKS(xc=xc, kpts=kpts))
        dat = krks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_j=True)
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            vj1 = FFTDF(cell1).get_jk(dm, kpts=cell1.make_kpts(kmesh), with_k=False)[0]
            vj1 *= .5
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            vj2 = FFTDF(cell2).get_jk(dm, kpts=cell2.make_kpts(kmesh), with_k=False)[0]
            vj2 *= .5
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            de = np.einsum('kij,kji', dm, (vj1-vj2)) / len(kpts)
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 2e-9

    def test_get_nuc(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(len(kmesh), nao, nao) - (.5+.2j)
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        kpts = cell.make_kpts(kmesh)
        mf_grad = krks.Gradients(cell.KRKS(xc=xc, kpts=kpts))
        dat = krks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_nuc=True)
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            vne1 = FFTDF(cell1).get_nuc(kpts=cell1.make_kpts(kmesh))
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            vne2 = FFTDF(cell2).get_nuc(kpts=cell2.make_kpts(kmesh))
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            de = np.einsum('kij,kji', dm, (vne1-vne2)) / len(kpts)
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 2e-9

    def test_get_pp(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='C 1 1 1; C 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]],
                     pseudo='gth-pade', a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(len(kmesh), nao, nao) - (.5+.1j)
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        kpts = cell.make_kpts(kmesh)
        mf_grad = krks.Gradients(cell.KRKS(xc=xc, kpts=kpts))
        dat = krks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_nuc=True)
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            vne1 = FFTDF(cell1).get_pp(kpts=cell1.make_kpts(kmesh))
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            vne2 = FFTDF(cell2).get_pp(kpts=cell2.make_kpts(kmesh))
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            de = np.einsum('kij,kji', dm, (vne1-vne2)) / len(kpts)
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
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).run()
        mf_grad = krks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

    def test_gga_vs_finite_difference(self):
        a = np.eye(3) * 3.5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='C 1 1 1; C 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     pseudo='gth-pade', a=a, unit='Bohr', verbose=0)
        xc = 'pbe'
        kmesh = [3, 1, 1]
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).run()
        mf_grad = krks.Gradients(mf)
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
        kmesh = [3, 1, 1]
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).run()
        mf_grad = krks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
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
        mf = krkspu.KRKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao)
        mf.__dict__.update(cell.KRKS(kpts=kpts).run(max_cycle=1).__dict__)
        sigma = krks_stress._hubbard_U_deriv1(mf)

        for (i, j) in [(1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            mf.reset(cell1)
            e1 = mf.get_veff().E_U.real
            mf.reset(cell2)
            e2 = mf.get_veff().E_U.real
            assert abs(sigma[i,j] - (e1 - e2) / 2e-4) < 1e-8

    def test_krkspu_finite_diff_high_cost(self):
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
        mf = krkspu.KRKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao).run()
        sigma = mf.Gradients().get_stress()
        mf_scanner = mf.as_scanner()

        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-3)
        e1 = mf_scanner(cell1)
        e2 = mf_scanner(cell2)
        assert abs(sigma[0,0] - (e1 - e2)/2e-3/cell.vol) < 1e-6

if __name__ == "__main__":
    print("Full Tests for KRKS Stress tensor")
    unittest.main()
