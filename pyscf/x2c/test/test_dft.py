#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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

import numpy
import unittest
from pyscf import gto
from pyscf import lib
from pyscf.x2c import x2c, dft
try:
    import mcfun
except ImportError:
    mcfun = None


class KnownValues(unittest.TestCase):
    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_vs_gks(self):
        with lib.temporary_env(lib.param, LIGHT_SPEED=20):
            mol = gto.M(atom='C', basis='6-31g')
            ref = dft.RKS(mol)
            ref.xc = 'pbe'
            ref.collinear = 'mcol'
            ref._numint.spin_samples = 6
            ref.run()

            c = numpy.vstack(mol.sph2spinor_coeff())
            mo1 = c.dot(ref.mo_coeff)
            dm = ref.make_rdm1(mo1, ref.mo_occ)
            mf = mol.GKS().x2c1e()
            mf.xc = 'pbe'
            mf.collinear = 'mcol'
            mf._numint.spin_samples = 6
            mf.max_cycle = 1
            mf.kernel(dm0=dm)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 10)


if __name__ == "__main__":
    print("Full Tests for X2C-KS")
    unittest.main()
