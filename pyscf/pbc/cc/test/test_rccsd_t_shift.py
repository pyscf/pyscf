#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Authors: Hong-Zhou Ye <hzyechem@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto, scf, cc
from pyscf.cc.ccsd_t import kernel as CCSD_T


def run_cell(cell, scaled_center):
    kpt = cell.make_kpts([1,1,1], scaled_center=scaled_center)[0]

    mf = scf.RHF(cell, kpt=kpt).rs_density_fit()
    mf.with_df.omega = 0.1
    mf.kernel()

    mcc = cc.RCCSD(mf)
    eris = mcc.ao2mo()
    mcc.kernel(eris=eris)
    eccsd = mcc.e_corr

    et = CCSD_T(mcc, eris)

    return eccsd, et


class KnownValues(unittest.TestCase):
    def test_water(self):
        atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        basis = 'gth-dzvp'
        pseudo = 'gth-hf-rev'
        a = np.eye(3) * 30
        cell = gto.M(atom=atom, basis=basis, a=a, pseudo=pseudo)

        eccsd_gamma, et_gamma = run_cell(cell, [0,0,0])
        self.assertAlmostEqual(eccsd_gamma, -0.2082317212, 8)
        self.assertAlmostEqual(et_gamma   , -0.0033716894, 8)

        eccsd_shifted, et_shifted = run_cell(cell, [0.1,0.1,0.1])
        self.assertAlmostEqual(eccsd_gamma, eccsd_shifted, 8)
        self.assertAlmostEqual(et_gamma   , et_shifted   , 8)

if __name__ == '__main__':
    print("RCCSD(T) with shift k-point test")
    unittest.main()
