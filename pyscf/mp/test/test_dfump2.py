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

        mmp = mp.dfump2.DFUMP2(mf, frozen=0)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.15321910903780495, 8)

        mmp = mp.dfump2.DFUMP2(mf, frozen=np.array([0]))
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, -0.15103334394544674, 8)

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
        ftmp = lib.NamedTemporaryFile()

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
        cell.spin = 1
        cell.charge = 1

        cell.basis = {'H': 'cc-pvdz',
                     'O': 'cc-pvdz',}
        cell.build()
        mf = scf.UHF(cell).density_fit()
        mf.conv_tol = 1e-12
        mf.scf()

        # incore using pre-cached CDERI
        mmp = mp.dfump2.DFUMP2(mf)
        mmp.kernel()
        eref = mmp.e_corr

        # direct MP2 starts here
        mf.with_df._cderi = None

        # incore
        mmp = mp.dfump2.DFUMP2(mf)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, eref, 8)

        # outcore
        mmp = mp.dfump2.DFUMP2(mf).set(force_outcore=True)
        mmp.kernel()
        self.assertAlmostEqual(mmp.e_corr, eref, 8)


    def test_dfump2_converted_rohf(self):
        # An ROHF reference converted to UHF is auto-detected as non-canonical
        # (f_ov != 0) and the T1 singles is included (issue #1687).
        molr = gto.M(atom='N 0 0 0; H 0 0 1.0; H 0.94 0 -0.33; H -0.94 0 -0.33',
                     charge=1, spin=1, basis='sto-3g', verbose=0)
        mf = scf.ROHF(molr).run(conv_tol=1e-12)
        pt = mp.dfump2.DFUMP2(mf.to_uhf().density_fit()).run()
        self.assertGreater(abs(pt.t1[0]).max(), 1e-3)
        self.assertAlmostEqual(pt.e_corr, -0.0408591463, 6)

    def test_dfump2_non_canonical(self):
        # Non-canonical reference: converged UHF orbitals rotated between the
        # occupied and virtual spaces, so that the occupied-virtual Fock block
        # is nonzero and the first-order singles (T1) contribute (issue #1687).
        # The same (identical) orbitals are used for the DF and 4-center
        # calculations, so the results agree within the fitting error.
        mol1 = gto.M(atom='O 0 0 0; O 0 0 1.2222', spin=2, basis='sto-3g',
                     verbose=0)
        mf1 = scf.UHF(mol1).run()
        nocca = numpy.count_nonzero(mf1.mo_occ[0] > 0)
        noccb = numpy.count_nonzero(mf1.mo_occ[1] > 0)
        ct, st = numpy.cos(0.4), numpy.sin(0.4)
        c = [x.copy() for x in mf1.mo_coeff]
        for s, nocc in [(0, nocca), (1, noccb)]:
            cocc, cvir = c[s][:,nocc-1].copy(), c[s][:,nocc].copy()
            c[s][:,nocc-1], c[s][:,nocc] = ct*cocc + st*cvir, -st*cocc + ct*cvir

        mfr = scf.UHF(mol1)
        mfr.__dict__.update(mf1.__dict__)
        mfr.mo_coeff = c
        mfr.converged = False
        pt = mp.UMP2(mfr).run(conv_tol=1e-10)

        mfd = scf.UHF(mol1).density_fit()
        mfd.__dict__.update(mf1.__dict__)
        mfd.mo_coeff = c
        mfd.converged = False
        mmp = mp.dfump2.DFUMP2(mfd)
        mmp.conv_tol = 1e-10
        mmp.kernel()
        self.assertTrue(mmp.converged)
        self.assertAlmostEqual(abs(mmp.e_corr - pt.e_corr), 0, 4)
        # t1 depends only on the Fock matrix, hence identical for DF and 4c
        self.assertAlmostEqual(max(abs(x - y).max() for x, y in
                                   zip(mmp.t1, pt.t1)), 0, 4)
        self.assertAlmostEqual(mmp.e_corr_singles, pt.e_corr_singles, 4)

if __name__ == "__main__":
    print("Full Tests for dfump2")
    unittest.main()
