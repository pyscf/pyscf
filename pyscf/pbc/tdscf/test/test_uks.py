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


class DiamondPBE(unittest.TestCase):
    ''' Reproduce RKS-TDSCF results
    '''
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

        xc = 'pbe'
        mf = scf.UKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend').run()
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

    def test_tda(self):
        ref = [4.77090949, 4.77090949]
        self.kernel('TDA', ref)

    def test_tdhf(self):
        ref = [4.7455726874, 4.7455726874]
        self.kernel('TDDFT', ref)


class WaterBigBoxPBE(unittest.TestCase):
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
        cell.spin = 2
        cell.a = np.eye(3) * 15
        cell.basis = 'sto-3g'
        cell.build()

        xc = 'pbe'
        mf = scf.UKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend')
        mf.with_df.omega = 0.1
        mf.kernel()
        cls.cell = cell
        cls.mf = mf

        mol = molgto.Mole()
        for key in ['verbose','output','atom','spin','basis']:
            setattr(mol, key, getattr(cell, key))
        mol.build()
        molmf = molscf.UKS(mol).set(xc=xc).density_fit(auxbasis=mf.with_df.auxbasis).run()
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
        self.assertTrue(abs(td.e[:self.nstates_test] * unitev -
                            moltd.e[:self.nstates_test] * unitev).max() < 0.1)

    def test_tda(self):
        self.kernel('TDA')

    def test_tdhf(self):
        self.kernel('TDDFT')


class DiamondPBE0(unittest.TestCase):
    ''' Reproduce RKS-TDSCF results
    '''
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

        xc = 'pbe0'
        mf = scf.UKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend').run()
        mf.grids.level = 4
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

    def test_tda(self):
        ref = [5.1069789, 5.10697989]
        self.kernel('TDA', ref)

    def test_tdhf(self):
        ref = [4.5331029147, 4.5331029147]
        self.kernel('TDDFT', ref)


class WaterBigBoxPBE0(unittest.TestCase):
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
        cell.spin = 2
        cell.a = np.eye(3) * 15
        cell.basis = 'sto-3g'
        cell.build()

        xc = 'pbe0'
        mf = scf.UKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend')
        mf.with_df.omega = 0.1
        mf.kernel()
        cls.cell = cell
        cls.mf = mf

        mol = molgto.Mole()
        for key in ['verbose','output','atom','spin','basis']:
            setattr(mol, key, getattr(cell, key))
        mol.build()
        molmf = molscf.UKS(mol).set(xc=xc).density_fit(auxbasis=mf.with_df.auxbasis).run()
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
        self.assertTrue(abs(td.e[:self.nstates_test] * unitev -
                            moltd.e[:self.nstates_test] * unitev).max() < 0.1)

    def test_tda(self):
        self.kernel('TDA')

    def test_tdhf(self):
        self.kernel('TDDFT')


if __name__ == "__main__":
    print("Full Tests for uks-TDA and uks-TDDFT")
    unittest.main()
