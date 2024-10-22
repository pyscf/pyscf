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
#
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import tempfile
import numpy as np

from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import kuhf
from pyscf.pbc.scf import kghf
from pyscf.pbc import df
import pyscf.pbc.tools

def make_primitive_cell(mesh):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh

    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    return cell

def setUpModule():
    global cell, kmf, kumf, kgmf, kpts
    cell = make_primitive_cell([9]*3)
    kpts = cell.make_kpts([3,1,1])
    kmf = khf.KRHF(cell, kpts, exxdiv='vcut_sph').run(conv_tol=1e-9)
    kumf = kuhf.KUHF(cell, kpts, exxdiv='vcut_sph').run(conv_tol=1e-9)
    kgmf = kghf.KGHF(cell, kpts, exxdiv='vcut_sph').run(conv_tol=1e-9)

def tearDownModule():
    global cell, kmf, kumf, kgmf
    cell.stdout.close()
    del cell, kmf, kumf, kgmf

class KnownValues(unittest.TestCase):
    def test_analyze(self):
        rpop, rchg = kmf.analyze()[0]
        upop, uchg = kumf.analyze()[0]
        gpop, gchg = kgmf.analyze()[0]
        self.assertTrue(isinstance(rpop, np.ndarray) and rpop.ndim == 1)
        self.assertAlmostEqual(abs(upop[0]+upop[1]-rpop).max(), 0, 7)
        self.assertAlmostEqual(abs(gpop[0]+gpop[1]-rpop).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(rpop), 1.638430, 5)

    def test_kpt_vs_supercell_high_cost(self):
        # For large n, agreement is always achieved
        # n = 17
        # For small n, agreement only achieved if "wrapping" k-k'+G in get_coulG
        n = 9
        nk = (3, 1, 1)
        cell = make_primitive_cell([n]*3)

        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        self.assertAlmostEqual(ekpt, -11.221426249047617, 8)

#        nk = (5, 1, 1)
#        abs_kpts = cell.make_kpts(nk, wrap_around=True)
#        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
#        ekpt = kmf.scf()
#        self.assertAlmostEqual(ekpt, -12.337299166550796, 8)

        supcell = pyscf.pbc.tools.super_cell(cell, nk)
        mf = pscf.RHF(supcell, exxdiv='vcut_sph')
        esup = mf.scf()/np.prod(nk)
        self.assertAlmostEqual(ekpt, esup, 8)

    def test_init_guess_by_chkfile(self):
        n = 9
        nk = (1, 1, 1)
        cell = make_primitive_cell([n]*3)

        kpts = cell.make_kpts(nk)
        kmf = khf.KRHF(cell, kpts, exxdiv='vcut_sph')
        kmf.chkfile = tempfile.NamedTemporaryFile().name
        kmf.conv_tol = 1e-9
        ekpt = kmf.scf()
        dm1 = kmf.make_rdm1()
        dm2 = kmf.from_chk(kmf.chkfile)
        self.assertTrue(dm2.dtype == np.double)
        self.assertTrue(np.allclose(dm1, dm2))

        mf = pscf.RHF(cell, exxdiv='vcut_sph')
        mf.chkfile = kmf.chkfile
        mf.init_guess = 'chkfile'
        mf.max_cycle = 1
        e1 = mf.kernel()
        mf.conv_check = False
        self.assertAlmostEqual(e1, ekpt, 9)

        nk = (3, 1, 1)
        kpts = cell.make_kpts(nk)
        kmf1 = khf.KRHF(cell, kpts, exxdiv='vcut_sph')
        kmf1.conv_tol = 1e-9
        kmf1.chkfile = mf.chkfile
        kmf1.init_guess = 'chkfile'
        kmf1.max_cycle = 2
        ekpt = kmf1.scf()
        kmf1.conv_check = False
        self.assertAlmostEqual(ekpt, -11.215259853108822, 8)

    def test_krhf(self):
        self.assertAlmostEqual(kmf.e_tot, -11.218735269838586, 8)

    def test_kuhf(self):
        self.assertAlmostEqual(kumf.e_tot, -11.218735269838586, 8)

        np.random.seed(1)
        kpts_bands = np.random.random((2,3))
        e = kumf.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(lib.fp(np.array(e)), -0.0455444, 6)

    def test_krhf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = np.eye(3) * 4,
                   mesh = [25,30,30],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = khf.KRHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.with_df.eta = 0.2
        mf.init_guess = 'hcore'
        mf.kpts = cell.make_kpts([2,1,1])
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.5113107, 6)

    def test_krhf_2d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = np.eye(3) * 4,
                   mesh = [25,25,40],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 2,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = khf.KRHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.with_df.eta = 0.2
        mf.with_df.mesh = cell.mesh
        mf.kpts = cell.make_kpts([2,1,1])
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.53769771, 4)

    def test_kuhf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = np.eye(3) * 4,
                   mesh = [25,40,40],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = kuhf.KUHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.with_df.eta = 0.2
        mf.init_guess = 'hcore'
        mf.kpts = cell.make_kpts([2,1,1])
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.5113107, 6)

    def test_kghf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = np.eye(3) * 4,
                   mesh = [25,40,40],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pscf.KGHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.with_df.eta = 0.2
        mf.init_guess = 'hcore'
        mf.kpts = cell.make_kpts([2,1,1])
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.5113107, 6)

    def test_get_fermi(self):
        self.assertAlmostEqual(kmf.get_fermi(), 0.33154831914017424, 6)

        def occ_vir(nocc, nvir):
            occ = np.zeros(nocc+nvir)
            occ[:nocc] = 1
            return occ
        mo_e_kpts = [np.arange(5), np.arange(2, 6)]
        mo_occ_kpts = [occ_vir(2, 3)*2, occ_vir(2, 2)*2]
        f = kmf.get_fermi(mo_e_kpts, mo_occ_kpts)
        self.assertAlmostEqual(f, 2, 9)

        # Smearing with error
        mo_occ_kpts[0][1:3] = 1.000001
        f = kmf.get_fermi(mo_e_kpts, mo_occ_kpts)
        self.assertAlmostEqual(f, 2, 9)

        mo_e_kpts = [mo_e_kpts, [x-.5 for x in mo_e_kpts]]
        mo_occ_kpts = [[occ_vir(3, 2), occ_vir(2, 2)],
                       [occ_vir(2, 3), occ_vir(1, 3)]]
        f = kumf.get_fermi(mo_e_kpts, mo_occ_kpts)
        self.assertAlmostEqual(f[0], 3, 9)
        self.assertAlmostEqual(f[1], 1.5, 9)

        # Smearing with error
        mo_occ_kpts[0][0][2:4] = 0.500001
        mo_occ_kpts[1][1][0] -= 0.0000001
        f = kumf.get_fermi(mo_e_kpts, mo_occ_kpts)
        self.assertAlmostEqual(f[0], 3, 9)
        self.assertAlmostEqual(f[1], 1.5, 9)

    def test_krhf_vs_rhf(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pscf.RHF(cell, k, exxdiv='vcut_sph')
        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()

        kmf = pscf.KRHF(cell, [k], exxdiv='vcut_sph')
        kmf.max_cycle = 1
        kmf.diis = None
        e2 = kmf.kernel()
        self.assertAlmostEqual(e1, e2, 9)
        self.assertAlmostEqual(e1, -11.451118801956275, 9)

    def test_small_system(self):
        mol = pbcgto.Cell(
            atom='He 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KRHF(mol,kpts=[[0., 0., 0.]])
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -2.2719576422665635, 8)

    def test_damping(self):
        nao = cell.nao
        np.random.seed(1)
        f = kmf.get_hcore()
        df  = np.random.rand(len(kpts),nao,nao)
        f_prev = f + df
        damp = 0.3
        f_damp = khf.get_fock(kmf, h1e=0, s1e=0, vhf=f, dm=0, cycle=0,
                              diis_start_cycle=2, damp_factor=damp, fock_last=f_prev)
        for k in range(len(kpts)):
            self.assertAlmostEqual(abs(f_damp[k] - (f[k]*(1-damp) + f_prev[k]*damp)).max(), 0, 9)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf")
    unittest.main()
