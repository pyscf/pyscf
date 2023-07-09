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
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = TD(self.mf).set(**kwargs).run()
        self.assertAlmostEqual(abs(td.e[0] * unitev  - ref).max(), 0, 4)

    def test_tda_singlet(self):
        ref = [9.6425852905]
        self.kernel(tdscf.TDA, ref)

    def test_tda_triplet(self):
        ref = [4.7209460257]
        self.kernel(tdscf.TDA , ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [9.2573219105]
        self.kernel(tdscf.TDHF , ref)

    def test_tdhf_triplet(self):
        ref = [3.0396052214]
        self.kernel(tdscf.TDHF , ref, singlet=False)


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
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(**kwargs).run()
        self.assertAlmostEqual(abs(td.e[0] * unitev  - ref).max(), 0, 4)

    def test_tda_singlet(self):
        ref = [12.7166510188]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [8.7359078688]
        self.kernel('TDA' , ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [12.6104811730]
        self.kernel('TDHF', ref)

    def test_tdhf_triplet(self):
        ref = [3.8940277713]
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

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.mol.stdout.close()
        del cls.cell, cls.mf
        del cls.mol, cls.molmf

    def kernel(self, TD, MOLTD, **kwargs):
        td = TD(self.mf).set(**kwargs).run()
        moltd = MOLTD(self.molmf).set(**kwargs).run()
        self.assertAlmostEqual(abs(td.e * unitev  - moltd.e * unitev).max(), 0, 2)

    def test_tda_singlet(self):
        self.kernel(tdscf.TDA , moltdscf.TDA)

    def test_tda_triplet(self):
        self.kernel(tdscf.TDA , moltdscf.TDA, singlet=False)

    def test_tdhf_singlet(self):
        self.kernel(tdscf.TDHF , moltdscf.TDHF)

    def test_tdhf_triplet(self):
        self.kernel(tdscf.TDHF , moltdscf.TDHF, singlet=False)


if __name__ == "__main__":
    print("Full Tests for rhf-TDA and rhf-TDHF")
    unittest.main()
