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
from pyscf.pbc.cc.eom_kccsd_rhf import EOMEESinglet
from pyscf.data.nist import HARTREE2EV as unitev


class DiamondPBE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        mf = scf.KRKS(cell, kpts=kpts).set(xc=xc).rs_density_fit(auxbasis='weigend').run()

        cls.cell = cell
        cls.mf = mf

        cls.nstates = 4 # make sure first `nstates_test` states are converged
        cls.nstates_test = 2
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(kshift_lst=np.arange(len(self.mf.kpts)),
                                        nstates=self.nstates, **kwargs).run()
        for kshift,e in enumerate(td.e):
            self.assertAlmostEqual(abs(e[:self.nstates_test] * unitev - ref[kshift]).max(), 0, 4)

    def test_tda_singlet(self):
        ref = [[7.7172854747, 7.7173219160],
               [8.3749594280, 8.3749980463]]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [[5.7465112548, 5.7465526327],
               [6.9888184993, 6.9888609925]]
        self.kernel('TDA', ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [[7.5824302393, 7.5824675688],
               [8.3438648420, 8.3439036271]]
        self.kernel('TDDFT', ref)

    def test_tdhf_triplet(self):
        ref = [[5.5659966435, 5.5660393021],
               [6.7992845776, 6.7993290046]]
        self.kernel('TDDFT', ref, singlet=False)


class DiamondPBE0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(kshift_lst=np.arange(len(self.mf.kpts)),
                                        nstates=self.nstates, **kwargs).run()
        for kshift,e in enumerate(td.e):
            self.assertAlmostEqual(abs(e[:self.nstates_test] * unitev - ref[kshift]).max(), 0, 4)

    def test_tda_singlet(self):
        ref = [[9.3936718451, 9.4874866060],
               [10.0697605303, 10.0697862958]]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [[6.6703797643, 6.6704110631],
               [7.4081863259, 7.4082204017]]
        self.kernel('TDA', ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [[9.2519208050, 9.3208025447],
               [9.9609751875, 9.9610015227]]
        self.kernel('TDDFT', ref)

    def test_tdhf_triplet(self):
        ref = [[6.3282716764, 6.3283051217],
               [7.0656766298, 7.0657111705]]
        self.kernel('TDDFT', ref, singlet=False)


if __name__ == "__main__":
    print("Full Tests for krks-TDA and krks-TDDFT")
    unittest.main()
