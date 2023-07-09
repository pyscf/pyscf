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
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, **kwargs):
        td = getattr(self.mf, TD)().set(kshift_lst=np.arange(len(self.mf.kpts)), **kwargs).run()
        for kshift,e in enumerate(td.e):
            self.assertAlmostEqual(abs(e * unitev  - ref[kshift]).max(), 0, 4)

    def test_tda_singlet(self):
        ref = [[7.7172937578, 7.7173147005, 8.1745659545],
               [8.3749670085, 8.3749891087, 9.8698173091]]
        self.kernel('TDA', ref)

    def test_tda_triplet(self):
        ref = [[5.7465210070, 5.7465445289, 6.0162087172],
               [6.9888275254, 6.9888517848, 7.0624984994]]
        self.kernel('TDA', ref, singlet=False)

    def test_tdhf_singlet(self):
        ref = [[7.5824389449, 7.5824603478, 7.9792278471],
               [8.3438724396, 8.3438964956, 9.5884734426]]
        self.kernel('TDDFT', ref)

    def test_tdhf_triplet(self):
        ref = [[5.5660080681, 5.5660310544, 5.9000406293],
               [6.7992898892, 6.7993190413, 6.8544503647]]
        self.kernel('TDDFT', ref, singlet=False)


if __name__ == "__main__":
    print("Full Tests for krks-TDA and krks-TDDFT")
    unittest.main()
