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
    nocc, nvir = a.shape[:2]
    a = a.reshape(nocc*nvir, -1)
    b = b.reshape(nocc*nvir, -1)
    h = np.block([[a        , b       ],
                  [-b.conj(),-a.conj()]])
    e = np.linalg.eigvals(np.asarray(h))
    lowest_e = np.sort(e[e.real > 0].real)
    lowest_e = lowest_e[lowest_e > 1e-3][:nroots]
    return lowest_e

class Diamond(unittest.TestCase):
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
        cell.pseudo = 'gth-pbe'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()

        xc = 'm06'
        mf = scf.RKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend').run()

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

    def test_tda_singlet(self):
        ref = [14.68587442, 14.68589929]
        td = self.kernel('TDA', ref)
        a, b = td.get_ab()
        no, nv = a.shape[:2]
        eref = np.linalg.eigvalsh(a.reshape(no*nv,-1))
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, 8)

    def test_tda_triplet(self):
        ref = [11.10049832, 11.59365532]
        self.kernel('TDA', ref, singlet=False)

    def test_tddft_singlet(self):
        ref = [14.42819773, 14.42822009]
        td = self.kernel('TDDFT', ref)
        a, b = td.get_ab()
        eref = diagonalize(a, b)
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, 8)

    def test_tddft_triplet(self):
        ref = [ 9.09496456, 11.53650896]
        self.kernel('TDDFT', ref, singlet=False)

    def check_rsh_tda(self, xc, place=6):
        cell = self.cell
        mf = cell.RKS(xc=xc).run()
        td = mf.TDA().run(nstates=5, conv_tol=1e-7)
        a, b = td.get_ab()
        no, nv = a.shape[:2]
        eref = np.linalg.eigvalsh(a.reshape(no*nv,-1))
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, place)

    def test_camb3lyp_tda(self):
        self.check_rsh_tda('camb3lyp')

    def test_wb97_tda(self):
        self.check_rsh_tda('wb97')

    @unittest.skip('HSE03 differs significantly between libxc-6.0 and 7.0, causing errors')
    def test_hse03_tda(self):
        self.check_rsh_tda('hse03')

    def test_hse06_tda(self):
        # reducing tol, as larger numerical uncertainties found in fxc when using libxc-7
        self.check_rsh_tda('hse06', place=3)


class DiamondPBEShifted(unittest.TestCase):
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
        cell.pseudo = 'gth-pbe'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()

        kpt = np.asarray([0.3721, 0.2077, 0.1415])

        xc = 'pbe'
        mf = scf.RKS(cell, kpt).set(xc=xc).rs_density_fit(auxbasis='weigend').run()

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

    def test_tda_singlet(self):
        ref = [11.9664870288, 12.7605699008]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [8.5705015296, 9.3030273411]
        self.kernel('TDA', ref, singlet=False)

    def test_tddft_singlet(self):
        ref = [11.8322851619, 12.6207316217]
        self.kernel('TDDFT', ref)

    def test_tddft_triplet(self):
        ref = [8.4227532516, 9.1695913993]
        self.kernel('TDDFT', ref, singlet=False)


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
        cell.a = np.eye(3) * 15
        cell.basis = 'sto-3g'
        cell.build()

        xc = 'pbe'
        mf = scf.RKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend')
        mf.with_df.omega = 0.1
        mf.kernel()
        cls.cell = cell
        cls.mf = mf

        mol = molgto.Mole()
        for key in ['verbose','output','atom','basis']:
            setattr(mol, key, getattr(cell, key))
        mol.build()
        molmf = molscf.RKS(mol).set(xc=xc).density_fit(auxbasis=mf.with_df.auxbasis).run()
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
        # larger tol (0.1 eV) potentially due to DFT integration error
        self.assertTrue((abs(td.e[:self.nstates_test] * unitev -
                             moltd.e[:self.nstates_test] * unitev).max() < 0.1))

    def test_tda_singlet(self):
        self.kernel('TDA')

    def test_tda_triplet(self):
        self.kernel('TDA', singlet=False)

    def test_tddft_singlet(self):
        self.kernel('TDDFT')

    def test_tddft_triplet(self):
        self.kernel('TDDFT', singlet=False)


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
        cell.pseudo = 'gth-pbe'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()

        xc = 'pbe0'
        mf = scf.RKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend').run()

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

    def test_tda_singlet(self):
        ref = [9.62238067, 9.62238067]
        td = self.kernel('TDA', ref)
        a, b = td.get_ab()
        no, nv = a.shape[:2]
        eref = np.linalg.eigvalsh(a.reshape(no*nv,-1))
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, 8)

    def test_tda_triplet(self):
        ref = [5.39995144, 5.39995144]
        self.kernel('TDA', ref, singlet=False)

    def test_tddft_singlet(self):
        ref = [9.26011401, 9.26011401]
        td = self.kernel('TDDFT', ref)
        a, b = td.get_ab()
        eref = diagonalize(a, b)
        self.assertAlmostEqual(abs(td.e[:2] - eref[:2]).max(), 0, 8)

    def test_tddft_triplet(self):
        ref = [4.68905023, 4.81439580]
        self.kernel('TDDFT', ref, singlet=False)


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
        cell.a = np.eye(3) * 15
        cell.basis = 'sto-3g'
        cell.build()

        xc = 'pbe0'
        mf = scf.RKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend')
        mf.with_df.omega = 0.1
        mf.kernel()
        cls.cell = cell
        cls.mf = mf

        mol = molgto.Mole()
        for key in ['verbose','output','atom','basis']:
            setattr(mol, key, getattr(cell, key))
        mol.build()
        molmf = molscf.RKS(mol).set(xc=xc).density_fit(auxbasis=mf.with_df.auxbasis).run()
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
        # larger tol (0.1 eV) potentially due to DFT integration error
        self.assertTrue((abs(td.e[:self.nstates_test] * unitev -
                             moltd.e[:self.nstates_test] * unitev).max() < 0.1))

    def test_tda_singlet(self):
        self.kernel('TDA')

    def test_tda_triplet(self):
        self.kernel('TDA', singlet=False)

    def test_tddft_singlet(self):
        self.kernel('TDDFT')

    def test_tddft_triplet(self):
        self.kernel('TDDFT', singlet=False)


if __name__ == "__main__":
    print("Full Tests for rks-TDA and rks-TDDFT")
    unittest.main()
