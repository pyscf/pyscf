# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

# Author: Yuhao Chen <chenyuhao@shu.edu.cn>

"""Guard missing AO integrals before unrestricted high-order CC."""

import copy
import unittest

import numpy as np

from pyscf import cc, gto, lib, scf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g',
                        spin=1, verbose=0)
        with lib.with_omp_threads(1):
            cls.mf = scf.UHF(cls.mol).run(conv_tol=1e-12)
        assert cls.mf.converged

    def test_missing_ao_integrals_rejected_for_multiple_electrons(self):
        mf = copy.copy(self.mf)
        mf._eri = None
        mycc = cc.UCCSDT(mf)
        with self.assertRaisesRegex(NotImplementedError, 'AO integrals|AO2MO'):
            mycc.ao2mo()

    def test_single_electron_without_ao_integrals(self):
        mol = gto.M(atom='H 0 0 0', basis='6-31g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        self.assertTrue(mf.converged)
        mf._eri = None
        mycc = cc.UCCSDT(mf)
        eris = mycc.ao2mo()
        for name in ('pppp', 'PPPP', 'pPpP'):
            np.testing.assert_array_equal(getattr(eris, name), 0.)
        mycc.kernel(eris=eris)
        self.assertTrue(mycc.converged)
        self.assertAlmostEqual(mycc.e_corr, 0., 12)


if __name__ == '__main__':
    unittest.main()
