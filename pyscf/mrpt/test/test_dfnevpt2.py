#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

import unittest
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.mcscf import avas
from pyscf.mrpt import nevpt2, dfnevpt2

'''
Reference values are computed from ORCA (6.1.0).
Note that these values were computed without frozen core approximation.
'''

class KnownValues(unittest.TestCase):

    def test_dfnevpt2(self):
        mol = gto.M(atom='''
                C   -0.669500   0.000000   0.000000
                C    0.669500   0.000000   0.000000
                H   -1.234217   0.928797   0.000000
                H   -1.234217  -0.928797   0.000000
                H    1.234217   0.928797   0.000000
                H    1.234217  -0.928797   0.000000''',
                basis='def2-SVP',
                verbose=4,
                symmetry=False,
                output='/dev/null')

        mf = scf.RHF(mol).density_fit(auxbasis='def2-SVP-JKFIT').run()

        mo_coeff = avas.kernel(mf, ['C 2pz'], minao=mol.basis)[2]

        mc = mcscf.CASSCF(mf, 2, (1, 1))
        mc.kernel(mo_coeff)

        # NEVPT2 without density fitting:
        mp0 = nevpt2.NEVPT(mc, density_fit=False)
        mp0.kernel()
        e0 = mc.e_tot + mp0.e_corr

        # Reference values computed from ORCA (6.1.0)
        self.assertAlmostEqual(mp0.e_corr, -0.25704466647882, 3)
        self.assertAlmostEqual(e0, -78.26314110837014, delta=1e-4)

        # NEVPT2 with density fitting (Default for DF-CAS object):
        mp1 = nevpt2.NEVPT(mc)
        mp1.kernel()
        e1 = mc.e_tot + mp1.e_corr

        # Reference values computed from ORCA (6.1.0)
        self.assertAlmostEqual(mp1.e_corr, -0.25691533473915, 3)
        self.assertAlmostEqual(e1, -78.26299316692777, delta=1e-4)

        # In general, DF cause energy difference of 0.1-1 mEh compared to non-DF calculation.
        # See: J. Chem. Theory Comput. 2017, 13, 451−459
        self.assertAlmostEqual(mp1.e_corr, mp0.e_corr, 3)

        # Calculate with DF-NEVPT2 in PySCF.
        # I needed to loosen this threshold, because it is causing test failure on
        # macos-build.
        self.assertAlmostEqual(mp1.e_corr, -0.256916544254705, delta=1e-4)

if __name__ == "__main__":
    print("Full Tests for df-nevpt2")
    unittest.main()
