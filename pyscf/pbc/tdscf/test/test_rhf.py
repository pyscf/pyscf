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
from pyscf import gto as molgto, scf as molscf, tdscf as moltdscf
from pyscf.pbc import gto, scf, tdscf
from pyscf.data.nist import HARTREE2EV as unitev


class Diamond(unittest.TestCase):
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

        mf = scf.RHF(cell).rs_density_fit(auxbasis='weigend').run()
        cls.cell = cell
        cls.mf = mf

        cls.nstates = 5 # make sure first `nstates_test` states are converged
        cls.nstates_test = 2

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(nstates=self.nstates, **kwargs).run()
        self.assertAlmostEqual(abs(td.e[:self.nstates_test] * unitev  - ref).max(), 0, 4)
        return td

    def test_tda_singlet(self):
        ref = [9.6425852906, 9.6425852906]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [4.7209460258, 5.6500725912]
        self.kernel('TDA', ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [9.2573219105, 9.2573219106]
        td = self.kernel('TDHF', ref)
        a, b = td.get_ab()
        no, nv = a.shape[:2]
        a = a.reshape(no*nv,-1)
        b = b.reshape(no*nv,-1)
        h = np.block([[a, b], [-b.conj(), -a.conj()]])
        eref = np.sort(np.linalg.eigvals(h).real)
        eref = eref[eref > 1e-3]
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, 5)

    def test_tdhf_triplet(self):
        ref = [3.0396052214, 3.0396052214]
        self.kernel('TDHF', ref, singlet=False)


class DiamondShifted(unittest.TestCase):
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

        kpt = np.asarray([0.3721, 0.2077, 0.1415])

        mf = scf.RHF(cell, kpt).rs_density_fit(auxbasis='weigend').run()
        cls.cell = cell
        cls.mf = mf

        cls.nstates = 5 # make sure first `nstates_test` states are converged
        cls.nstates_test = 2

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(nstates=self.nstates, **kwargs).run()
        self.assertAlmostEqual(abs(td.e[:self.nstates_test] * unitev  - ref).max(), 0, 4)
        return td

    def test_tda_singlet(self):
        ref = [12.7166510188, 13.5460934688]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [8.7359078688, 9.2565904604]
        self.kernel('TDA' , ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [12.6104811733, 13.4160717812]
        td = self.kernel('TDHF', ref)
        a, b = td.get_ab()
        no, nv = a.shape[:2]
        a = a.reshape(no*nv,-1)
        b = b.reshape(no*nv,-1)
        h = np.block([[a, b], [-b.conj(), -a.conj()]])
        eref = np.sort(np.linalg.eigvals(h).real)
        eref = eref[eref > 1e-3]
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, 5)

    def test_tdhf_triplet(self):
        ref = [3.8940277713, 7.9448161493]
        self.kernel('TDHF', ref, singlet=False)


class WaterBigBox(unittest.TestCase):
    ''' Match molecular CIS
    '''
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        cell.a = np.eye(3) * 15
        cell.basis = 'sto-3g'
        cell.build()
        mf = scf.RHF(cell).rs_density_fit(auxbasis='weigend')
        mf.with_df.omega = 0.1
        mf.kernel()
        cls.cell = cell
        cls.mf = mf

        mol = molgto.Mole()
        for key in ['verbose','output','atom','basis']:
            setattr(mol, key, getattr(cell, key))
        mol.build()
        molmf = molscf.RHF(mol).density_fit(auxbasis=mf.with_df.auxbasis).run()
        cls.mol = mol
        cls.molmf = molmf

        cls.nstates = 5 # make sure first `nstates_test` states are converged
        cls.nstates_test = 2

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.mol.stdout.close()
        del cls.cell, cls.mf
        del cls.mol, cls.molmf

    def kernel(self, TD, **kwargs):
        td = getattr(self.mf, TD)().set(nstates=self.nstates, **kwargs).run()
        moltd = getattr(self.mf, TD)().set(nstates=self.nstates, **kwargs).run()
        self.assertAlmostEqual(abs(td.e[:self.nstates_test] * unitev -
                                   moltd.e[:self.nstates_test] * unitev).max(), 0, 2)

    def test_tda_singlet(self):
        self.kernel('TDA')

    def test_tda_triplet(self):
        self.kernel('TDA', singlet=False)

    def test_tdhf_singlet(self):
        self.kernel('TDHF')

    def test_tdhf_triplet(self):
        self.kernel('TDHF', singlet=False)


if __name__ == "__main__":
    print("Full Tests for rhf-TDA and rhf-TDHF")
    unittest.main()
