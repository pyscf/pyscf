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
from pyscf.gto import ATOM_OF, intor_cross
from pyscf.pbc import dft, gto, grad
from pyscf.pbc.tools import pbc
from pyscf.pbc.grad import rks_stress
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
        ngrids = np.prod(cell.mesh)
        w0, w1 = rks_stress._get_weight_strain_derivatives(cell, ngrids)
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

if __name__ == "__main__":
    print("Full Tests for RKS Stress tensor")
    unittest.main()
