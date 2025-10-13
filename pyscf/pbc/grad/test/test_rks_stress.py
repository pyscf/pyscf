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
from pyscf.pbc.dft.numint import NumInt
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.grad import rks_stress, rks
from pyscf.pbc.grad.rks_stress import _finite_diff_cells

def setUpModule():
    global cell
    a = np.eye(3) * 5
    np.random.seed(5)
    a += np.random.rand(3, 3) - .5
    cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                 basis=[[0, [.5, 1]], [1, [.5, 1]]], a=a, unit='Bohr')

class KnownValues(unittest.TestCase):
    def test_ovlp(self):
        sigma9 = rks_stress.get_ovlp(cell)
        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-5)
        s1 = cell1.pbc_intor('int1e_ovlp')
        s2 = cell2.pbc_intor('int1e_ovlp')
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[0] - ref).max() < 1e-9

        cell1, cell2 = _finite_diff_cells(cell, 0, 1, disp=1e-5)
        s1 = cell1.pbc_intor('int1e_ovlp')
        s2 = cell2.pbc_intor('int1e_ovlp')
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[1] - ref).max() < 1e-9

        aoslices = cell.aoslice_by_atom()
        ao_repeats = aoslices[:,3] - aoslices[:,2]
        bas_coords = np.repeat(cell.atom_coords(), ao_repeats, axis=0)
        ovlp10 = cell.pbc_intor('int1e_ipovlp')
        ovlp10 = np.einsum('xij,iy->xyij', ovlp10, bas_coords)

        nao = cell.nao
        Ls = cell.get_lattice_Ls()
        scell = cell.copy()
        scell = pbc._build_supcell_(scell, cell, Ls)
        sc_ovlp01 = intor_cross('int1e_ipovlp', scell, cell)
        sc_ovlp01 = sc_ovlp01.transpose(0,2,1).reshape(3,nao,-1,nao)
        ovlp01 = np.einsum('xinj,njy->xyij', sc_ovlp01, bas_coords+Ls[:,None])
        dat = -(ovlp10 + ovlp01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

        sc_ovlp10 = intor_cross('int1e_ipovlp', cell, scell)
        sc_ovlp10 = sc_ovlp10.reshape(3,nao,-1,nao)
        ovlp01 = np.einsum('xinj,njy->xyij', sc_ovlp10, bas_coords+Ls[:,None])
        dat = -(ovlp10 - ovlp01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

    def test_kin(self):
        sigma9 = rks_stress.get_kin(cell)
        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-5)
        s1 = cell1.pbc_intor('int1e_kin')
        s2 = cell2.pbc_intor('int1e_kin')
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[0] - ref).max() < 1e-9

        cell1, cell2 = _finite_diff_cells(cell, 0, 1, disp=1e-5)
        s1 = cell1.pbc_intor('int1e_kin')
        s2 = cell2.pbc_intor('int1e_kin')
        ref = (s1 - s2) / 2e-5
        assert abs(sigma9[1] - ref).max() < 1e-9

        aoslices = cell.aoslice_by_atom()
        ao_repeats = aoslices[:,3] - aoslices[:,2]
        bas_coords = np.repeat(cell.atom_coords(), ao_repeats, axis=0)
        kin10 = cell.pbc_intor('int1e_ipkin')
        kin10 = np.einsum('xij,iy->xyij', kin10, bas_coords)

        nao = cell.nao
        Ls = cell.get_lattice_Ls()
        scell = cell.copy()
        scell = pbc._build_supcell_(scell, cell, Ls)
        sc_kin01 = intor_cross('int1e_ipkin', scell, cell)
        sc_kin01 = sc_kin01.transpose(0,2,1).reshape(3,nao,-1,nao)
        kin01 = np.einsum('xinj,njy->xyij', sc_kin01, bas_coords+Ls[:,None])
        dat = -(kin10 + kin01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

        sc_kin10 = intor_cross('int1e_ipkin', cell, scell)
        sc_kin10 = sc_kin10.reshape(3,nao,-1,nao)
        kin01 = np.einsum('xinj,njy->xyij', sc_kin10, bas_coords+Ls[:,None])
        dat = -(kin10 - kin01)[0, 1]
        assert abs(dat - ref).max() < 1e-9

    def test_weight(self):
        grids = UniformGrids(cell)
        ngrids = grids.size
        w0, w1 = rks_stress._get_weight_strain_derivatives(cell, grids)
        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-5)
        assert abs(w1[0,0] - (cell1.vol - cell2.vol)/ngrids / 2e-5).max() < 1e-9
        cell1, cell2 = _finite_diff_cells(cell, 0, 1, disp=1e-5)
        assert abs(w1[0,1] - (cell1.vol - cell2.vol)/ngrids / 2e-5).max() < 1e-9

    def test_coulG(self):
        coulG0, coulG1 = rks_stress._get_coulG_strain_derivatives(cell, cell.Gv)
        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-5)
        assert abs(coulG1[0,0] - (pbc.get_coulG(cell1) - pbc.get_coulG(cell2)) / 2e-5).max() < 1e-9
        cell1, cell2 = _finite_diff_cells(cell, 0, 1, disp=1e-5)
        assert abs(coulG1[0,1] - (pbc.get_coulG(cell1) - pbc.get_coulG(cell2)) / 2e-5).max() < 1e-9

    def test_eval_ao_cart(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [3, [.7, 1]]], a=a, unit='Bohr', cart=True)
        coords = np.random.rand(10, 3)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords)[0]
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao(cell1, coords)
            ao2 = dft.numint.eval_ao(cell2, coords)
            assert abs(ao_value[i,j,0] - (ao1 - ao2) / 2e-5).max() < 1e-9

    def test_eval_ao_sph(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [3, [.7, 1]]],
                     a=a, unit='Bohr')
        coords = np.random.rand(10, 3)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords)[0]
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao(cell1, coords)
            ao2 = dft.numint.eval_ao(cell2, coords)
            assert abs(ao_value[i,j,0] - (ao1 - ao2) / 2e-5).max() < 1e-9

    def test_eval_ao_deriv1_cart(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [3, [.7, 1]]],
                     a=a, unit='Bohr', cart=True)
        coords = np.random.rand(10, 3)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords, deriv=1)[0]
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao(cell1, coords, deriv=1)
            ao2 = dft.numint.eval_ao(cell2, coords, deriv=1)
            assert abs(ao_value[i,j] - (ao1 - ao2) / 2e-5).max() < 1e-9

    def test_eval_ao_deriv1_sph(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [3, [.7, 1]]],
                     a=a, unit='Bohr')
        coords = np.random.rand(10, 3)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords, deriv=1)[0]
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao(cell1, coords, deriv=1)
            ao2 = dft.numint.eval_ao(cell2, coords, deriv=1)
            assert abs(ao_value[i,j] - (ao1 - ao2) / 2e-5).max() < 1e-9

    def test_eval_ao_grid_response(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]]], a=a, unit='Bohr', cart=True)

        mesh = [6,5,4]
        coords = cell.get_uniform_grids(mesh)
        ao = dft.numint.eval_ao(cell, coords, deriv=1)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords)[0]
        ao_value[:,:,0] += np.einsum('xgi,gy->xygi', ao[1:4], coords)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao(cell1, cell1.get_uniform_grids(mesh))
            ao2 = dft.numint.eval_ao(cell2, cell2.get_uniform_grids(mesh))
            assert abs(ao_value[i,j,0] - (ao1 - ao2) / 2e-5).max() < 1e-9

        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]]], a=a, unit='Bohr')

        mesh = [6,5,4]
        coords = cell.get_uniform_grids(mesh)
        ao = dft.numint.eval_ao(cell, coords, deriv=2)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords, deriv=1)[0]
        ao_value[:,:,0] += np.einsum('xgi,gy->xygi', ao[1:4], coords)
        ao_value[:,:,1] += np.einsum('xgi,gy->xygi', ao[4:7], coords)
        ao_value[0,:,2] += np.einsum('gi,gy->ygi', ao[5], coords) # YX
        ao_value[1,:,2] += np.einsum('gi,gy->ygi', ao[7], coords) # YY
        ao_value[2,:,2] += np.einsum('gi,gy->ygi', ao[8], coords) # YZ
        ao_value[0,:,3] += np.einsum('gi,gy->ygi', ao[6], coords) # ZX
        ao_value[1,:,3] += np.einsum('gi,gy->ygi', ao[8], coords) # ZY
        ao_value[2,:,3] += np.einsum('gi,gy->ygi', ao[9], coords) # ZZ
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            ao1 = dft.numint.eval_ao(cell1, cell1.get_uniform_grids(mesh), deriv=1)
            ao2 = dft.numint.eval_ao(cell2, cell2.get_uniform_grids(mesh), deriv=1)
            assert abs(ao_value[i,j] - (ao1 - ao2) / 2e-5).max() < 1e-9

    def test_lattice_vector_derivatives(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]]], a=a, unit='Bohr')
        ref = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                ref[i,j,:,i] = a[:,j]
        a1 = np.eye(3)[:,None,None] * a.T[:,:,None]
        assert abs(ref - a1).max() < 1e-9
        for i in range(3):
            for j in range(3):
                cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
                assert abs(a1[i,j] - (cell1.lattice_vectors() -
                                      cell2.lattice_vectors())/2e-5).max() < 1e-9

    def test_get_vxc_lda(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc))
        dat = rks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_vxc_gga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'pbe,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc))
        dat = rks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_vxc_mgga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'm06,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc))
        dat = rks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_j(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc))
        dat = rks_stress.get_vxc(mf_grad, cell, dm, with_j=True)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            vj1 = FFTDF(cell1).get_jk(dm, with_k=False)[0]
            vj1 *= .5
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            vj2 = FFTDF(cell2).get_jk(dm, with_k=False)[0]
            vj2 *= .5
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('ij,ji', dm, (vj1-vj2))
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 2e-9

    def test_get_nuc(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc))
        dat = rks_stress.get_vxc(mf_grad, cell, dm, with_nuc=True)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            vne1 = FFTDF(cell1).get_nuc()[0]
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            vne2 = FFTDF(cell2).get_nuc()[0]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('ij,ji', dm, (vne1-vne2))
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 2e-9

    def test_get_pp(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='Si 1 1 1; C 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]],
                     pseudo='gth-pade', a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc))
        dat = rks_stress.get_vxc(mf_grad, cell, dm, with_nuc=True)
        ni = NumInt()
        kpt = np.zeros(3)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            vne1 = FFTDF(cell1).get_pp(kpt)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            vne2 = FFTDF(cell2).get_pp(kpt)
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('ij,ji', dm, (vne1-vne2))
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 1e-8

    def test_lda_vs_finite_difference(self):
        a = np.eye(3) * 3
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='H 1 1 1; H 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     a=a, unit='Bohr', verbose=0)
        mf = cell.RKS(xc='svwn').run()
        mf_grad = rks.Gradients(mf)
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
        mf = cell.RKS(xc='pbe').run()
        mf_grad = rks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
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
        mf = cell.RKS(xc='rscan').run()
        mf_grad = rks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

if __name__ == "__main__":
    print("Full Tests for RKS Stress tensor")
    unittest.main()
