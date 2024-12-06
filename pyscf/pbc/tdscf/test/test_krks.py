#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from pyscf import __config__
from pyscf.pbc import gto, scf, tdscf, cc
from pyscf import gto as molgto, scf as molscf, tdscf as moltdscf
from pyscf.dft import radi
from pyscf.pbc.cc.eom_kccsd_rhf import EOMEESinglet
from pyscf.data.nist import HARTREE2EV as unitev


def diagonalize(a, b, nroots=4):
    nkpts, nocc, nvir = a.shape[:3]
    a = a.reshape(nkpts*nocc*nvir, -1)
    b = b.reshape(nkpts*nocc*nvir, -1)
    h = np.block([[a        , b       ],
                  [-b.conj(),-a.conj()]])
    e = np.linalg.eigvals(np.asarray(h))
    lowest_e = np.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e

class DiamondPBE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
        cell.a = '''
        1.7850000000 1.7850000000 0.0000000000
        0.0000000000 1.7850000000 1.7850000000
        1.7850000000 0.0000000000 1.7850000000
        '''
        cell.pseudo = 'gth-hf-rev'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()
        kpts = cell.make_kpts((2,1,1))

        xc = 'pbe'
        mf = scf.KRKS(cell, kpts=kpts).set(xc=xc).rs_density_fit(auxbasis='weigend')
        mf.with_df._j_only = False
        mf.with_df.build()
        mf.run()

        cls.cell = cell
        cls.mf = mf

        cls.nstates = 4 # make sure first `nstates_test` states are converged
        cls.nstates_test = 2

    @classmethod
    def tearDownClass(cls):
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(kshift_lst=np.arange(len(ref)),
                                        nstates=self.nstates, **kwargs).run()
        for kshift,e in enumerate(td.e):
            self.assertAlmostEqual(abs(e[:self.nstates_test] * unitev - ref[kshift]).max(), 0, 5)
        return td

    def test_tda_singlet(self):
        ref = [[7.7172857896, 7.7173222336]]
        td = self.kernel('TDA', ref)
        a0, _ = td.get_ab(kshift=0)
        nk, no, nv = a0.shape[:3]
        eref = np.linalg.eigvalsh(a0.reshape(nk*no*nv,-1))[:4]
        self.assertAlmostEqual(abs(td.e[0][:2] - eref[:2]).max(), 0, 7)

    def test_tda_triplet(self):
        ref = [[5.7465112548, 5.7465526327]]
        self.kernel('TDA', ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [[7.58243026, 7.58246786]]
        td = self.kernel('TDDFT', ref)
        a, b = td.get_ab(kshift=0)
        eref = diagonalize(a, b)
        self.assertAlmostEqual(abs(td.e[0][:2] - eref[:2]).max(), 0, 7)

    def test_tdhf_triplet(self):
        ref = [[5.56599665, 5.56603980]]
        self.kernel('TDDFT', ref, singlet=False)


class DiamondPBE0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
        cell.a = '''
        1.7850000000 1.7850000000 0.0000000000
        0.0000000000 1.7850000000 1.7850000000
        1.7850000000 0.0000000000 1.7850000000
        '''
        cell.pseudo = 'gth-hf-rev'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()
        kpts = cell.make_kpts((2,1,1))

        xc = 'pbe0'
        mf = scf.KRKS(cell, kpts=kpts).set(xc=xc).rs_density_fit(auxbasis='weigend').run()

        cls.cell = cell
        cls.mf = mf

        cls.nstates = 5 # make sure first `nstates_test` states are converged
        cls.nstates_test = 2

    @classmethod
    def tearDownClass(cls):
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(kshift_lst=np.arange(len(ref)),
                                        nstates=self.nstates, **kwargs).run()
        for kshift,e in enumerate(td.e):
            self.assertAlmostEqual(abs(e[:self.nstates_test] * unitev - ref[kshift]).max(), 0, 5)
        return td

    def test_tda_singlet(self):
        ref = [[9.3936718451, 9.4874866060]]
        td = self.kernel('TDA', ref)
        a0, _ = td.get_ab(kshift=0)
        nk, no, nv = a0.shape[:3]
        eref = np.linalg.eigvalsh(a0.reshape(nk*no*nv,-1))[:4]
        self.assertAlmostEqual(abs(td.e[0][:2] - eref[:2]).max(), 0, 8)

    def test_tda_triplet(self):
        ref = [[6.6703797643, 6.6704110631]]
        self.kernel('TDA', ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [[9.25192096, 9.32080304]]
        td = self.kernel('TDDFT', ref, conv_tol=1e-6)
        a, b = td.get_ab(kshift=0)
        eref = diagonalize(a, b)
        self.assertAlmostEqual(abs(td.e[0][:2] - eref[:2]).max(), 0, 8)

    def test_tdhf_triplet(self):
        ref = [[6.3282716764, 6.3283051217]]
        self.kernel('TDDFT', ref, singlet=False)


if __name__ == "__main__":
    print("Full Tests for krks-TDA and krks-TDDFT")
    unittest.main()
