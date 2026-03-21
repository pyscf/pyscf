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

import unittest
import tempfile
import numpy as np
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import kuhf
from pyscf.pbc.tools.pbc import super_cell

def setUpModule():
    global cell, mf, kmf, kpts
    cell = pgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = '321g'
    cell.a = np.eye(3) * 3
    cell.mesh = [8] * 3
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.spin = 2
    cell.build()
    nk = [2, 2, 1]
    kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pscf.KUHF(cell, kpts).run(conv_tol=1e-8)
    mf = pscf.UHF(cell).run(conv_tol=1e-8)

def tearDownModule():
    global cell, kmf, mf
    cell.stdout.close()
    del cell, kmf, mf

class KnownValues(unittest.TestCase):
    def test_kuhf_kernel(self):
        self.assertAlmostEqual(kmf.e_tot, -4.594854184081046, 8)
        e4 = super_cell(cell, [2,2,1]).KUHF().run().e_tot
        self.assertAlmostEqual(kmf.e_tot - e4/4, 0, 8)
        kmf.analyze()

    def test_uhf_kernel(self):
        self.assertAlmostEqual(mf.e_tot, -3.3634535013441855, 8)
        mf.analyze()

    def test_kuhf_vs_uhf(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pscf.UHF(cell, k, exxdiv='vcut_sph')
        dm = mf.get_init_guess(key='1e')

        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel(dm)

        nao = cell.nao
        kmf = pscf.KUHF(cell, [k], exxdiv='vcut_sph')
        kmf.max_cycle = 1
        kmf.diis = None
        e2 = kmf.kernel(dm.reshape(2,1,nao,nao))
        self.assertAlmostEqual(e1, e2, 9)
        self.assertAlmostEqual(e1, -3.498612316383892, 8)

    def test_init_guess_by_chkfile(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pscf.KUHF(cell, [k], exxdiv='vcut_sph')
        mf.chkfile = tempfile.NamedTemporaryFile().name
        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.4070772194665477, 7)

        mf1 = pscf.UHF(cell, exxdiv='vcut_sph')
        mf1.chkfile = mf.chkfile
        mf1.init_guess = 'chkfile'
        mf1.diis = None
        mf1.max_cycle = 1
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -3.4272925247351256, 7)
        self.assertTrue(mf1.mo_coeff[0].dtype == np.double)

    @unittest.skip('mesh not enough for density')
    def test_dipole_moment(self):
        dip = mf.dip_moment()
        self.assertAlmostEqual(abs(dip).max(), 0, 2)

        dip = kmf.dip_moment()
        self.assertAlmostEqual(abs(dip).max(), 0, 2)

    def test_spin_square(self):
        ss = kmf.spin_square()[0]
        self.assertAlmostEqual(ss, 2.0836508842313273, 4)

    def test_bands(self):
        np.random.seed(1)
        kpts_bands = np.random.random((1,3))

        e = mf.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(lib.fp(e), 0.8857024, 5)

        e = kmf.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(lib.fp(e), -0.309626, 5)

    def test_small_system(self):
        mol = pgto.Cell(
            atom='H 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            spin=1,
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -0.10439957735616917, 8)

        mol = pgto.Cell(
            atom='He 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -2.2719576422665635, 8)

    def test_invalid_occupancy(self):
        cell = pgto.M(a=np.eye(3)*5.,
                      atom='He 0 0 1',
                      basis=[[0, [.6, 1]]], spin=2)
        mf = cell.KUHF(kpts=cell.make_kpts([2,1,1]))
        self.assertRaises(RuntimeError, mf.run)

if __name__ == '__main__':
    print("Tests for PBC UHF and PBC KUHF")
    unittest.main()
