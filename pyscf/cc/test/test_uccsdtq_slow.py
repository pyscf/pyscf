# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Yuhao Chen <chenyuhao@shu.edu.cn>

"""Expensive complete-solve UCCSDTQ regression; run explicitly.

python -m pytest -o addopts=--import-mode=importlib pyscf/cc/test/test_uccsdtq_slow.py
"""

from math import sqrt
import unittest

import numpy as np
from pyscf import cc, gto, lib, scf


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.previous_threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.previous_threads)

    def test_eight_correlated_electron_ch4_restricted_limit(self):
        a = 1.09 / sqrt(3)
        atom = [('C', (0, 0, 0))] + [
            ('H', tuple(a * x for x in signs)) for signs in
            ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
        mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-13)
        expected = cc.RCCSDTQ(mf, frozen=1).run(conv_tol=1e-12,
                                              conv_tol_normt=1e-10, max_cycle=100)
        seed = cc.UCCSDT(mf.to_uhf(), frozen=1, compact_tamps=True).run(
            conv_tol=1e-11, conv_tol_normt=1e-9, max_cycle=100)
        solver = cc.UCCSDTQ(mf.to_uhf(), frozen=1)
        solver.quadruples_backend = 'blocked'
        solver.conv_tol, solver.conv_tol_normt = 1e-11, 1e-9
        solver.max_cycle = 100
        solver.kernel(tamps=seed.tamps)
        self.assertTrue(expected.converged and seed.converged and solver.converged)
        self.assertEqual(solver.nocc, (4, 4))
        self.assertEqual(solver.nmo, (8, 8))
        self.assertEqual(tuple(t.size for t in solver.t4), (1, 256, 1296, 256, 1))
        self.assertGreater(sum(np.linalg.norm(t) for t in solver.t4), 1e-5)
        self.assertAlmostEqual(solver.e_tot, expected.e_tot, 9)


if __name__ == '__main__':
    unittest.main()
