# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from pyscf import lib
from pyscf import scf
from pyscf.pbc import gto as pbcgto

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = pbcgto.M(
            a = np.array([
                [2, 0, 0],
                [0, 1.5, 0],
                [0, 0, 2.],
            ]),
            atom = """
                H 0 0 0
                H 1.1 0.1 0
            """,
            basis = """
                H  DZV-GTH-q1 DZV-GTH
                2
                1  0  0  4  2
                    8.3744350009  -0.0283380461   0.0000000000
                    1.8058681460  -0.1333810052   0.0000000000
                    0.4852528328  -0.3995676063   0.0000000000
                    0.1658236932  -0.5531027541   1.0000000000
                # 2  1  1  1  1
                #     0.7270000000   1.0000000000
                1 1 1 1 1
                    0.08  1.0
            """, # This is the gth-dzv basis
            verbose = 7,
            precision = 1e-8,
            output = '/dev/null',
        )
        cls.cell = cell

        cls.kpts = cell.make_kpts([3,2,1])

        cls.backup = scf.hf.remove_overlap_zero_eigenvalue
        scf.hf.remove_overlap_zero_eigenvalue = True

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        scf.hf.remove_overlap_zero_eigenvalue = cls.backup

    def test_rks(self):
        cell = self.cell
        mf = cell.RKS(xc = "PBE")
        mf.conv_tol = 1e-10
        mf = mf.multigrid_numint()
        mf.max_cycle=1
        test_energy = mf.kernel()
        ref_energy = -1.7809196678928856
        assert abs(test_energy - ref_energy) < 2e-10

    def test_krks(self):
        cell = self.cell
        kpts = self.kpts
        mf = cell.KRKS(kpts = kpts, xc = "PBE")
        mf.conv_tol = 1e-10
        mf = mf.multigrid_numint()
        test_energy = mf.kernel()
        ref_energy = -1.153344217311621
        assert abs(test_energy - ref_energy) < 2e-10

        e, c = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
        assert abs(e[e<1e7] - mf.mo_energy[e<1e7]).max() < 5e-7
        f = mf.get_fock()
        e1 = lib.einsum('kpi,kpq,kqi->ki', c.conj(), f, c)
        assert abs(e[e<1e7] - e1[e<1e7]).max() < 2e-10

    def test_uks(self):
        cell = self.cell
        mf = cell.UKS(xc = "LDA")
        mf.conv_tol = 1e-10
        mf = mf.multigrid_numint()
        test_energy = mf.kernel()
        ref_energy = -1.6790448390335038
        assert abs(test_energy - ref_energy) < 2e-10

    def test_kuks(self):
        cell = self.cell
        kpts = self.kpts
        mf = cell.KUKS(kpts = kpts, xc = "LDA")
        mf = mf.multigrid_numint()
        mf.conv_tol = 1e-10
        test_energy = mf.kernel()
        ref_energy = -1.0410239448726277
        assert abs(test_energy - ref_energy) < 2e-10

        e, c = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
        assert abs(e[e<1e7] - mf.mo_energy[e<1e7]).max() < 5e-7
        f = mf.get_fock()
        e1 = lib.einsum('skpi,skpq,skqi->ski', c.conj(), f, c)
        assert abs(e[e<1e7] - e1[e<1e7]).max() < 2e-10

if __name__ == '__main__':
    print("Full Tests for PBC with diffuse orbitals")
    unittest.main()
