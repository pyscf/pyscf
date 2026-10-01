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
from functools import reduce
import numpy
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mp
from pyscf import df

def setUpModule():
    global mol, mf, dfmf, dfmf2
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    dfmf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    dfmf.conv_tol = 1e-12
    dfmf.kernel()

    dfmf2 = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    dfmf2.conv_tol = 1e-12
    dfmf2.kernel()

def tearDownModule():
    global mol, mf, dfmf, dfmf2
    mol.stdout.close()
    del mol, mf, dfmf, dfmf2


class KnownValues(unittest.TestCase):
    def test_dfmp2_direct(self):
        # incore
        mmp = mp.dfmp2.DFMP2(mf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.20400482102770082, 8)

        # outcore
        mmp = mp.dfmp2.DFMP2(mf).set(force_outcore=True)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.20400482102770082, 8)

    def test_dfmp2_frozen(self):
        mmp = mp.dfmp2.DFMP2(mf, frozen=[0,1,5])
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.13844381496025246, 8)

        mmp = mp.dfmp2.DFMP2(mf, frozen=0)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.2040048210277004, 8)

        mmp = mp.dfmp2.DFMP2(mf, frozen=np.array([0]))
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.20166760413156876, 8)

    def test_dfmp2_mf_with_df(self):
        mmpref = mp.mp2.MP2(dfmf)
        mmpref.kernel()

        mmp = mp.dfmp2.DFMP2(dfmf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, mmpref.e_corr, 8)
        self.assertAlmostEqual(abs(mmp.t2-mmpref.t2).max(), 0, 8)

        mmp = mp.dfmp2.DFMP2(dfmf).set(force_outcore=True)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, mmpref.e_corr, 8)
        self.assertAlmostEqual(abs(mmp.t2-mmpref.t2).max(), 0, 8)

    def test_dfmp2_mf_with_df_diff_auxbasis(self):
        mp2_df = df.DF(mol)
        mp2_df.auxbasis = 'cc-pvdz-ri'
        mmp0 = mp.dfmp2.DFMP2(dfmf2)
        mmp0.with_df = mp2_df
        mmp0.kernel()
        ref_e_corr = mmp0.e_corr
        self.assertAlmostEqual(ref_e_corr, -0.20399004345216082, 8)

        mmp1 = mp.MP2(dfmf2).density_fit(auxbasis='cc-pvdz-ri')
        mmp1.kernel()
        self.assertAlmostEqual(mmp1.e_corr, ref_e_corr, 8)

        mmp2 = mp.dfmp2.DFMP2(dfmf2)
        mmp2.with_df.auxbasis = 'cc-pvdz-ri'
        mmp2.kernel()
        self.assertAlmostEqual(mmp2.e_corr, ref_e_corr, 8)

    def test_read_ovL_incore(self):
        mmp = mp.dfmp2.DFMP2(mf)
        eris = mmp.ao2mo()
        mmp.kernel(eris=eris)

        mmp1 = mp.dfmp2.DFMP2(mf)
        eris = mmp1.ao2mo(ovL=eris.ovL)
        mmp1.kernel(eris=eris)

        self.assertAlmostEqual(mmp.e_corr, mmp1.e_corr, 8)

    def test_read_ovL_outcore(self):
        ftmp = lib.NamedTemporaryFile()

        mmp = mp.dfmp2.DFMP2(mf)
        eris = mmp.ao2mo(ovL_to_save=ftmp.name)
        mmp.kernel(eris=eris)

        mmp1 = mp.dfmp2.DFMP2(mf)
        eris = mmp1.ao2mo(ovL=ftmp.name)
        mmp1.kernel(eris=eris)

        self.assertAlmostEqual(mmp.e_corr, mmp1.e_corr, 8)

    def test_dfmp2_slow(self):
        from pyscf.mp import dfmp2_slow
        # incore
        mmp = dfmp2_slow.DFMP2(mf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.20400482102770082, 8)

    def test_dfmp2_pbc(self):
        from pyscf.pbc import gto, scf
        cell = gto.Cell()
        cell.verbose = 7
        cell.output = '/dev/null'
        cell.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        cell.a = numpy.eye(3) * 5

        cell.basis = {'H': 'cc-pvdz',
                     'O': 'cc-pvdz',}
        cell.build()
        mf = scf.RHF(cell).density_fit()
        mf.conv_tol = 1e-12
        mf.scf()

        # incore using pre-cached CDERI
        mmp = mp.dfmp2.DFMP2(mf)
        mmp.kernel()
        eref = mmp.e_corr

        # direct MP2 starts here
        mf.with_df._cderi = None

        # incore
        mmp = mp.dfmp2.DFMP2(mf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, eref, 8)

        # outcore
        mmp = mp.dfmp2.DFMP2(mf).set(force_outcore=True)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, eref, 8)


    def test_dfmp2_non_canonical(self):
        # Non-canonical reference: converged HF orbitals rotated between the
        # occupied and virtual spaces, so that the occupied-virtual Fock block
        # is nonzero and the first-order singles (T1) contribute (issue #1687).
        # The same (identical) orbitals are used for the DF and 4-center
        # calculations, so the results agree within the fitting error.
        mol1 = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 0.24', basis='sto-3g',
                     verbose=0)
        mf1 = scf.RHF(mol1).run()
        nocc = mol1.nelectron // 2
        ct, st = numpy.cos(0.4), numpy.sin(0.4)
        c = mf1.mo_coeff.copy()
        cocc, cvir = c[:,nocc-1].copy(), c[:,nocc].copy()
        c[:,nocc-1], c[:,nocc] = ct*cocc + st*cvir, -st*cocc + ct*cvir

        mfr = scf.RHF(mol1)
        mfr.__dict__.update(mf1.__dict__)
        mfr.mo_coeff = c
        mfr.converged = False
        pt = mp.MP2(mfr).run(conv_tol=1e-10)

        mfd = scf.RHF(mol1).density_fit()
        mfd.__dict__.update(mf1.__dict__)
        mfd.mo_coeff = c
        mfd.converged = False
        mmp = mp.dfmp2.DFMP2(mfd)
        mmp.conv_tol = 1e-10
        mmp.kernel()
        self.assertTrue(mmp.converged)
        self.assertAlmostEqual(abs(mmp.e_corr - pt.e_corr), 0, 4)
        # t1 depends only on the Fock matrix, hence identical for DF and 4c
        self.assertAlmostEqual(abs(mmp.t1 - pt.t1).max(), 0, 4)
        self.assertAlmostEqual(mmp.e_corr_singles, pt.e_corr_singles, 4)

if __name__ == "__main__":
    print("Full Tests for dfmp2")
    unittest.main()
