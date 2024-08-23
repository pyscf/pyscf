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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Chia-Nan Yeh <yehcanon@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import dft
from pyscf.pbc import tools
from pyscf.pbc.x2c import sfx2c1e, x2c1e

def setUpModule():
    global cell, cell1
    cell = gto.Cell()
    cell.build(unit = 'B',
               a = numpy.eye(3)*4,
               mesh = [11]*3,
               atom = 'H 0 0 0; H 0 0 1.8',
               verbose = 0,
               basis='sto3g')
    cell1 = gto.Cell()
    cell1.atom = '''
    He   1.3    .2       .3
    He    .1    .1      1.1 '''
    cell1.basis = {'He': [[0, [0.8, 1]],
                          [1, [0.6, 1]]
                         ]}
    cell1.mesh = [15] * 3
    cell1.a = numpy.array(([2.0,  .9, 0. ],
                           [0.1, 1.9, 0.4],
                           [0.8, 0  , 2.1]))
    cell1.build()

def tearDownModule():
    global cell, cell1
    del cell, cell1

class KnownValues(unittest.TestCase):
    def test_hf(self):
        with lib.light_speed(4) as c:
            mf = scf.RHF(cell1).sfx2c1e()
            mf.with_df = df.AFTDF(cell1)
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.33795518021098797 + 0j, 8)
            e = mf.kernel()
            self.assertAlmostEqual(e, -4.757961450353405, 8)

            mf.with_x2c.approx = 'ATOM1E'
            kpts = cell1.make_kpts([3,1,1])
            h1 = mf.get_hcore(kpt=kpts[1])
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.32361715420090226 + 0j, 8)

    def test_hf_high_cost(self):
        with lib.light_speed(2) as c:
            mf = scf.RHF(cell).sfx2c1e()
            mf.with_df = df.AFTDF(cell)
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.20522508213548604 + 0j, 8)
            kpts = cell.make_kpts([3, 1, 1])
            h1 = mf.get_hcore(kpt=kpts[1])
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.004971818990083491 + 0j, 8)
            e = mf.kernel()
            self.assertAlmostEqual(e, -1.701698627990108, 8)

            mf.with_x2c.approx = 'ATOM1E'
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.2458227312351979+0j, 8)
            kpts = cell.make_kpts([3,1,1])
            h1 = mf.get_hcore(kpt=kpts[1])
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.04113247191600125+0j, 8)

    def test_khf_high_cost(self):
        with lib.light_speed(2) as c:
            mf = scf.KRHF(cell).sfx2c1e()
            mf.with_df = df.AFTDF(cell)
            mf.kpts = cell.make_kpts([3,1,1])
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]), -0.25949615001885146 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]), -0.006286599310207025 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]), -0.006286599310170855 + 0j, 8)
            e = mf.kernel()
            self.assertAlmostEqual(e, -1.47623306917771, 8)

            mf.with_x2c.approx = 'ATOM1E'
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]), -0.31082970748083477+0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]), -0.05200981271862468+0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]), -0.05200981271862468+0j, 8)

    def test_kghf_high_cost(self):
        with lib.light_speed(2) as c:
            # KGHF.sfx2c1e should reproduce the KRHF.x2c1e() result
            mf = scf.KGHF(cell).sfx2c1e()
            mf.with_df = df.AFTDF(cell)
            mf.kpts = cell.make_kpts([3, 1, 1])
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]), -0.2594961500188514 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]), -0.006286599310207025 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]), -0.006286599310170855 + 0j, 8)
            e = mf.kernel()
            self.assertAlmostEqual(e, -1.47623306917771, 8)

            mf = scf.KGHF(cell).x2c1e()
            mf.with_df = df.AFTDF(cell)
            mf.kpts = cell.make_kpts([3, 1, 1])
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]), -0.2594961500188514 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]), -0.006285311085086392 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]), -0.006285311085049715 + 0j, 8)

            e = mf.kernel()
            self.assertAlmostEqual(e, -1.4762329948110864, 8)

    def test_khf_hcore(self):
        with lib.light_speed(4) as c:
            mf = scf.KRHF(cell1).sfx2c1e()
            mf.with_df = df.AFTDF(cell1)
            mf.kpts = cell1.make_kpts([3,1,1])
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]), -0.34134301670830286 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]), -0.10997040473326099 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]), -0.10997040473320772 + 0j, 8)

    def test_kghf_hcore(self):
        with lib.light_speed(4) as c:
            mf = scf.KGHF(cell1).x2c1e()
            mf.with_df = df.AFTDF(cell1)
            mf.kpts = cell1.make_kpts([3, 1, 1])
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]), -0.3413265650564443 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]), -0.1099177448595494 + 0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]), -0.1099177448595494 + 0j, 8)

    def test_pnucp(self):
        charge = -cell1.atom_charges()
        Gv = cell1.get_Gv(cell1.mesh)
        SI = cell1.get_SI(Gv)
        rhoG = numpy.dot(charge, SI)

        coulG = tools.get_coulG(cell1, mesh=cell1.mesh, Gv=Gv)
        vneG = rhoG * coulG
        vneR = tools.ifft(vneG, cell1.mesh).real

        coords = cell1.gen_uniform_grids(cell1.mesh)
        aoR = dft.numint.eval_ao(cell1, coords, deriv=1)
        ngrids, nao = aoR.shape[1:]
        vne_ref = numpy.einsum('p,xpi,xpj->ij', vneR, aoR[1:4], aoR[1:4])

        mydf = df.AFTDF(cell1)
        dat = sfx2c1e.get_pnucp(mydf)
        self.assertAlmostEqual(abs(dat-vne_ref).max(), 0, 6)

    def test_pvxp(self):
        # w_soc should not depend on eta and the type of DF class
        kpts = cell1.make_kpts([3, 1, 1])
        mydf = df.AFTDF(cell1, kpts=kpts[1])
        #dat = x2c1e.get_pbc_pvxp_legacy(cell1, kpts[1])
        ref = dat = x2c1e.get_pbc_pvxp(mydf, kpts[1])
        self.assertAlmostEqual(dat[0].sum(), 0.0 + -0.11557054307865766j, 7)
        self.assertAlmostEqual(dat[1].sum(), 0.0 + -0.19650430913542424j, 7)
        self.assertAlmostEqual(dat[2].sum(), 0.0 + 0.25706456053958415j, 7)

        # GDF
        mydf = df.GDF(cell1)
        mydf.kpts = kpts
        mydf.build()
        dat = x2c1e.get_pbc_pvxp(mydf, kpts[1])
        self.assertAlmostEqual(abs(dat - ref).max(), 0, 6)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.x2c")
    unittest.main()
