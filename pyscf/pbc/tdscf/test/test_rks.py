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
setattr(__config__, 'tdscf_rhf_TDDFT_deg_eia_thresh', 1e-1)         # make sure no missing roots
from pyscf import gto as molgto, scf as molscf, tdscf as moltdscf
from pyscf.pbc import gto, scf, tdscf
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
        cell.pseudo = 'gth-pbe'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()

        xc = 'pbe'
        mf = scf.RKS(cell).set(xc=xc).rs_density_fit(auxbasis='weigend').run()

        cls.cell = cell
        cls.mf = mf
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(**kwargs).run()
        self.assertAlmostEqual(abs(td.e * unitev  - ref).max(), 0, 5)

    def test_tda_singlet(self):
        ref = [9.2717239608, 9.2717239608, 9.2717425470]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [4.7947342558, 4.7947342558, 4.7947605634]
        self.kernel('TDA', ref, singlet=False)

    def test_tddft_singlet(self):
        ref = [8.8773591552, 8.8773591552, 8.8773808758]
        self.kernel('TDDFT', ref)

    def test_tddft_triplet(self):
        ref = [4.7694603180, 4.7694603180, 4.7694891742]
        self.kernel('TDDFT', ref, singlet=False)


class DiamondPBEShifted(unittest.TestCase):
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
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(**kwargs).run()
        self.assertAlmostEqual(abs(td.e * unitev  - ref).max(), 0, 5)

    def test_tda_singlet(self):
        ref = [11.9664896841, 12.7605720987, 15.1738260142]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [8.5705050199, 9.3030310573, 11.4378496190]
        self.kernel('TDA', ref, singlet=False)

    def test_tddft_singlet(self):
        # TODO: these do not match pbc.tdscf.rhf.TDHF(mf)
        ref = [10.5406839130, 11.2987220401, 13.1534367448]
        self.kernel('TDDFT', ref)

    def test_tddft_triplet(self):
        ref = [9.8870284077, 10.5054352535, 10.9880745420]
        self.kernel('TDDFT', ref, singlet=False)


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

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.mol.stdout.close()
        del cls.cell, cls.mf
        del cls.mol, cls.molmf

    def kernel(self, TD, **kwargs):
        td = getattr(self.mf, TD)().set(**kwargs).run()
        moltd = getattr(self.molmf, TD)().set(**kwargs).run()
        # larger tol (0.1 eV) potentially due to DFT integration error
        self.assertTrue((abs(td.e * unitev  - moltd.e * unitev).max() < 0.1))

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
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(**kwargs).run()
        self.assertAlmostEqual(abs(td.e * unitev  - ref).max(), 0, 5)

    def test_tda_singlet(self):
        ref = [9.6154699758, 9.6154699758, 9.6154827819]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [5.1302206493, 5.1302206493, 5.1302404747]
        self.kernel('TDA', ref, singlet=False)

    def test_tddft_singlet(self):
        ref = [9.2585544813, 9.2585544813, 9.2585695773]
        self.kernel('TDDFT', ref)

    def test_tddft_triplet(self):
        ref = [4.5570175405, 4.5570175405, 4.5662667978]
        self.kernel('TDDFT', ref, singlet=False)


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

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.mol.stdout.close()
        del cls.cell, cls.mf
        del cls.mol, cls.molmf

    def kernel(self, TD, **kwargs):
        td = getattr(self.mf, TD)().set(**kwargs).run()
        moltd = getattr(self.molmf, TD)().set(**kwargs).run()
        # larger tol (0.1 eV) potentially due to DFT integration error
        self.assertTrue((abs(td.e * unitev  - moltd.e * unitev).max() < 0.1))

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
