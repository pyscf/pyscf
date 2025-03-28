#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
import tempfile
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mp

def setUpModule():
    global mol, mf, dfmf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.spin = 1
    mol.charge = 1

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    dfmf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    dfmf.conv_tol = 1e-12
    dfmf.kernel()

def tearDownModule():
    global mol, mf, dfmf
    mol.stdout.close()
    del mol, mf, dfmf


class KnownValues(unittest.TestCase):
    def test_dfmp2_direct(self):
        # incore
        mmp = mp.dfump2.DFUMP2(mf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.15321910903780497, 8)

        # outcore
        mmp = mp.dfump2.DFUMP2(mf).set(force_outcore=True)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.15321910903780497, 8)

    def test_dfmp2_frozen(self):
        mmp = mp.dfump2.DFUMP2(mf, frozen=[[0,1,5], [1]])
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.09397152054462676, 8)

    def test_dfmp2_mf_with_df(self):
        mmpref = mp.ump2.UMP2(dfmf)
        mmpref.kernel()

        mmp = mp.dfump2.DFUMP2(dfmf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, mmpref.e_corr, 8)
        for s in [0,1,2]:
            self.assertAlmostEqual(abs(mmp.t2[s]-mmpref.t2[s]).max(), 0, 8)

        mmp = mp.dfump2.DFUMP2(dfmf).set(force_outcore=True)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, mmpref.e_corr, 8)
        for s in [0,1,2]:
            self.assertAlmostEqual(abs(mmp.t2[s]-mmpref.t2[s]).max(), 0, 8)

    def test_read_ovL_incore(self):
        mmp = mp.dfump2.DFUMP2(mf)
        eris = mmp.ao2mo()
        mmp.kernel(eris=eris)

        mmp1 = mp.dfump2.DFUMP2(mf)
        eris = mmp1.ao2mo(ovL=eris.ovL)
        mmp1.kernel(eris=eris)

        self.assertAlmostEqual(mmp.e_corr, mmp1.e_corr, 8)

    def test_read_ovL_outcore(self):
        ftmp = tempfile.NamedTemporaryFile()

        mmp = mp.dfump2.DFUMP2(mf)
        eris = mmp.ao2mo(ovL_to_save=ftmp.name)
        mmp.kernel(eris=eris)

        mmp1 = mp.dfump2.DFUMP2(mf)
        eris = mmp1.ao2mo(ovL=ftmp.name)
        mmp1.kernel(eris=eris)

        self.assertAlmostEqual(mmp.e_corr, mmp1.e_corr, 8)

    def test_dfmp2_slow(self):
        from pyscf.mp import dfump2_slow
        # incore
        mmp = dfump2_slow.DFUMP2(mf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.15321910903780497, 8)


if __name__ == "__main__":
    print("Full Tests for dfump2")
    unittest.main()
