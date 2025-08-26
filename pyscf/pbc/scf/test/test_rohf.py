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
from pyscf.pbc.scf import krohf
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
    kmf = pscf.KROHF(cell, kpts).run()
    mf = pscf.ROHF(cell).run()

def tearDownModule():
    global cell, kmf, mf
    cell.stdout.close()
    del cell, kmf, mf

class KnownValues(unittest.TestCase):
    def test_krohf_kernel(self):
        self.assertAlmostEqual(kmf.e_tot, -4.57655196508766, 8)
        kmf.analyze()
        e4 = super_cell(cell, [2,2,1]).KROHF().run().e_tot
        self.assertAlmostEqual(kmf.e_tot - e4/4, 0, 8)

    def test_rohf_kernel(self):
        self.assertAlmostEqual(mf.e_tot, -3.3633746534777718, 8)
        mf.analyze()

    def test_krhf_vs_rhf(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pscf.ROHF(cell, k, exxdiv='vcut_sph')
        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()

        kmf = pscf.KROHF(cell, [k], exxdiv='vcut_sph')
        kmf.max_cycle = 1
        kmf.diis = None
        e2 = kmf.kernel()
        self.assertAlmostEqual(e1, e2, 9)
        self.assertAlmostEqual(e1, -2.9369005352340434, 8)

    def test_init_guess_by_chkfile(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pscf.KROHF(cell, [k], exxdiv='vcut_sph')
        mf.chkfile = tempfile.NamedTemporaryFile().name
        mf.init_guess = 'hcore'
        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.4376090968645068, 7)

        mf1 = pscf.ROHF(cell, exxdiv='vcut_sph')
        mf1.chkfile = mf.chkfile
        mf1.init_guess = 'chkfile'
        mf1.diis = None
        mf1.max_cycle = 1
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -3.4190632006601662, 7)
        self.assertTrue(mf1.mo_coeff[0].dtype == np.double)

    @unittest.skip('mesh not enough for density')
    def test_dipole_moment(self):
        dip = mf.dip_moment()
        self.assertAlmostEqual(abs(dip).max(), 0, 2)

        dip = kmf.dip_moment()
        self.assertAlmostEqual(abs(dip).max(), 0, 2)

    def test_get_init_guess(self):
        cell1 = cell.copy()
        cell1.dimension = 1
        cell1.low_dim_ft_type = 'inf_vacuum'
        cell1.build(0, 0)
        mf = pscf.ROHF(cell1)
        dm = mf.get_init_guess(key='minao')
        self.assertAlmostEqual(lib.fp(dm), -0.13837729124284337, 8)

        mf = pscf.KROHF(cell1)
        dm = mf.get_init_guess(key='minao')
        self.assertAlmostEqual(lib.fp(dm), -0.13837729124284337, 8)

    def test_spin_square(self):
        ss = kmf.spin_square()[0]
        self.assertAlmostEqual(ss, 2, 9)

    def test_analyze(self):
        pop, chg = kmf.analyze()[0]
        self.assertAlmostEqual(lib.fp(pop), 1.1514919154737624, 3)
        self.assertAlmostEqual(sum(chg), 0, 7)
        self.assertAlmostEqual(lib.fp(chg), -0.04683923436982078, 3)

    def test_small_system(self):
        # issue #686
        mol = pgto.Cell(
            atom='H 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            spin=1,
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KROHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -0.10439957735616917, 8)

        mol = pgto.Cell(
            atom='He 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KROHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -2.2719576422665635, 8)

if __name__ == '__main__':
    print("Tests for PBC ROHF and PBC KROHF")
    unittest.main()
