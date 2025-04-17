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
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import tempfile
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell
    L = 4.
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.a = np.eye(3)*L
    cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
    cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
    cell.build()

def tearDownModule():
    global cell
    del cell


class KnownValues(unittest.TestCase):
#    def test_pp_RKS(self):
#        cell = pbcgto.Cell()
#
#        cell.unit = 'A'
#        cell.atom = '''
#            Si    0.000000000    0.000000000    0.000000000;
#            Si    0.000000000    2.715348700    2.715348700;
#            Si    2.715348700    2.715348700    0.000000000;
#            Si    2.715348700    0.000000000    2.715348700;
#            Si    4.073023100    1.357674400    4.073023100;
#            Si    1.357674400    1.357674400    1.357674400;
#            Si    1.357674400    4.073023100    4.073023100;
#            Si    4.073023100    4.073023100    1.357674400
#        '''
#        cell.basis = 'gth-szv'
#        cell.pseudo = 'gth-pade'
#
#        Lx = Ly = Lz = 5.430697500
#        cell.a = np.diag([Lx,Ly,Lz])
#        cell.mesh = np.array([21]*3)
#
#        cell.verbose = 5
#        cell.output = '/dev/null'
#        cell.build()
#
#        mf = pbcdft.RKS(cell)
#        mf.xc = 'lda,vwn'
#        self.assertAlmostEqual(mf.scf(), -31.081616722101646, 8)


    def test_chkfile_k_point(self):
        cell = pbcgto.Cell()
        cell.a = np.eye(3) * 6
        cell.mesh = [21]*3
        cell.unit = 'B'
        cell.atom = '''He     2.    2.       3.
                      He     3.    2.       3.'''
        cell.basis = {'He': 'sto3g'}
        cell.verbose = 0
        cell.build()
        mf1 = pbcdft.RKS(cell)
        mf1.chkfile = tempfile.NamedTemporaryFile().name
        mf1.max_cycle = 1
        mf1.kernel()

        cell = pbcgto.Cell()
        cell.a = np.eye(3) * 6
        cell.mesh = [41]*3
        cell.unit = 'B'
        cell.atom = '''He     2.    2.       3.
                       He     3.    2.       3.'''
        cell.basis = {'He': 'ccpvdz'}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.nimgs = [2,2,2]
        cell.build()
        mf = pbcdft.RKS(cell)
        np.random.seed(10)
        mf.kpt = np.random.random(3)
        mf.max_cycle = 1
        dm = mf.from_chk(mf1.chkfile)
        mf.conv_check = False
        self.assertAlmostEqual(mf.scf(dm), -4.7090816314173365, 8)

    def test_density_fit(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['He' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.build()
        mf = pbcdft.RKS(cell).density_fit()
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -4.717699891018736, 6)

    def test_density_fit_2d(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.a[2,2] = 12
        cell.dimension = 2
        cell.unit = 'B'
        cell.atom = 'H 0 0 0; H .8 .8 0'
        cell.basis = {'H': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.build()
        mf = pbcdft.RKS(cell).run()
        self.assertAlmostEqual(mf.e_tot, -0.6252695697315944, 7)
        mf = pbcdft.RKS(cell).density_fit().run()
        self.assertAlmostEqual(mf.e_tot, -0.635069614773985, 5)

    def test_rsh_fft(self):
        mf = pbcdft.RKS(cell)
        mf.xc = 'hse06'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.482418296326724, 7)

        mf.xc = 'camb3lyp'
        mf.omega = .15
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.476617717375184, 7)

    @unittest.skip('TODO: Check other packages how exxdiv=vcut_sph is handled for RSH')
    def test_rsh_fft_vcut_sph(self):
        # Adding this test to ensure that the new SR treatment in get_veff is
        # compatible with the treatment (full-range - LR) in pyscf-2.7.
        # However, the results of HSE with exxdiv=vcut_sph might not be reasonable.
        mf = pbcdft.RKS(cell)
        mf.xc = 'hse06'
        mf.exxdiv = 'vcut_sph'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4319699945616375, 7)

    def test_custom_rsh_df(self):
        mf = pbcdft.RKS(cell).density_fit()
        mf.xc = 'wb97'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4916945546399165, 6)

        mf.xc = 'camb3lyp'
        mf.omega = .15
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4766238116030683, 6)

    def test_rsh_mdf(self):
        mf = pbcdft.RKS(cell).mix_density_fit()
        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4745138538438827, 6)

        mf.omega = .15
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4766174820185456, 6)

    def test_rsh_aft_high_cost(self):
        from pyscf.pbc.df.aft import AFTDF
        mf = pbcdft.RKS(cell)
        mf.with_df = AFTDF(cell)
        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4745140705800446, 7)

    def test_rsh_0d(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.verbose = 0
        cell.a = np.eye(3)*L
        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.dimension = 0
        cell.mesh = [60]*3
        cell.build()
        mf = pbcdft.RKS(cell).density_fit()
        mf.xc = 'camb3lyp'
        mf.omega = '0.7'
        mf.exxdiv = None
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4836186361124617, 3)

        mol = cell.to_mol()
        mf1 = mol.RKS().density_fit()
        mf1.xc = 'camb3lyp'
        mf1.omega = '0.7'
        mf1.kernel()
        self.assertAlmostEqual(mf1.e_tot-mf1.energy_nuc(), mf.e_tot-mf.energy_nuc(), 7)

    @unittest.skip('ewald should not be enabled for 0d')
    def test_rsh_0d_ewald(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.verbose = 0
        cell.a = np.eye(3)*L
        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.dimension = 0
        cell.mesh = [60]*3
        cell.build()
        mf = pbcdft.RKS(cell).density_fit()
        mf.xc = 'camb3lyp'
        mf.omega = '0.7'
        mf.exxdiv = 'ewald'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.47559566263186, 4)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()
