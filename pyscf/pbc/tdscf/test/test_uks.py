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
from pyscf.dft import radi
from pyscf.pbc import gto, scf, tdscf
from pyscf.data.nist import HARTREE2EV as unitev


def diagonalize(a, b, nroots=4):
    a = spin_orbital_block(a)
    b = spin_orbital_block(b, True)
    abba = np.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e = np.linalg.eig(abba)[0]
    lowest_e = np.sort(e[e.real > 0].real)
    lowest_e = lowest_e[lowest_e > 1e-3][:nroots]
    return lowest_e

def spin_orbital_block(a, symmetric=False):
    a_aa, a_ab, a_bb = a
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    a_ab = a_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    if symmetric:
        a_ba = a_ab.T
    else:
        a_ba = a_ab.conj().T
    a_bb = a_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    a = np.block([[a_aa, a_ab],
                  [a_ba, a_bb]])
    return a


class DiamondM06(unittest.TestCase):
    ''' Reproduce RKS-TDSCF results
    '''
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

        xc = 'm06'
        mf = scf.UKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend').run()
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
        td = getattr(self.mf, TD)().set(nstates=self.nstates, **kwargs).run()
        self.assertAlmostEqual(abs(td.e[:self.nstates_test] * unitev  - ref).max(), 0, 5)
        return td

    def test_tda(self):
        ref = [11.09731427, 11.57079413]
        td = self.kernel('TDA', ref)
        a, b = td.get_ab()
        a = spin_orbital_block(a)
        eref = np.linalg.eigvalsh(a)
        self.assertAlmostEqual(abs(td.e[:4] - eref[:4]).max(), 0, 8)

    def test_tdhf(self):
        ref = [9.09165361, 11.51362009]
        td = self.kernel('TDDFT', ref, conv_tol=1e-8)
        a, b = td.get_ab()
        eref = diagonalize(a, b)
        self.assertAlmostEqual(abs(td.e[:4] - eref[:4]).max(), 0, 8)

    def check_rsh_tda(self, xc, place=6):
        cell = self.cell
        mf = cell.UKS(xc=xc).run()
        td = mf.TDA().run(nstates=3, conv_tol=1e-7)
        a, b = td.get_ab()
        a = spin_orbital_block(a)
        eref = np.linalg.eigvalsh(a)
        self.assertAlmostEqual(abs(td.e[:3] - eref[:3]).max(), 0, place)

    def test_camb3lyp_tda(self):
        self.check_rsh_tda('camb3lyp')

    def test_wb97_tda(self):
        self.check_rsh_tda('wb97')

    def test_hse03_tda(self):
        self.check_rsh_tda('hse03')


class WaterBigBoxPBE(unittest.TestCase):
    ''' Match molecular CIS
    '''
    @classmethod
    def setUpClass(cls):
        cls.original_grids = radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

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
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids
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

        xc = 'pbe0'
        mf = scf.UKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend').run()
        mf.grids.level = 4
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
        td = getattr(self.mf, TD)().set(nstates=self.nstates, **kwargs).run()
        self.assertAlmostEqual(abs(td.e[:self.nstates_test] * unitev  - ref).max(), 0, 5)
        return td

    def test_tda(self):
        ref = [5.37745381, 5.37745449]
        td = self.kernel('TDA', ref)
        a, b = td.get_ab()
        a = spin_orbital_block(a)
        eref = np.linalg.eigvalsh(a)
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, 8)

    def test_tdhf(self):
        # nstates=6 is required to derive the lowest state
        ref = [4.6851639, 4.79043398, 4.79043398]
        td = self.kernel('TDDFT', ref[1:3])
        a, b = td.get_ab()
        eref = diagonalize(a, b)
        self.assertAlmostEqual(abs(td.e[:2] - eref[1:3]).max(), 0, 8)


class WaterBigBoxPBE0(unittest.TestCase):
    ''' Match molecular CIS
    '''
    @classmethod
    def setUpClass(cls):
        cls.original_grids = radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

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
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids
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
