# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#
import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc import scf,adc,mp
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell

cell = gto.M(
    unit='B',
    a=[[0.,          3.37013733,  3.37013733],
         [3.37013733,  0.,          3.37013733],
         [3.37013733,  3.37013733,  0.        ]],
    mesh=[13]*3,
    atom='''He 0 0 0
              He 1.68506866 1.68506866 1.68506866''',
    basis='gth-dzv',
    pseudo='gth-pade',
    verbose=0,
)

nmp = [1,1,2]

class KnownValues(unittest.TestCase):
    def test_check_mol_mp2(self):
        # treating supercell at gamma point
        supcell = super_cell(cell,nmp)
        mf  = scf.RHF(supcell,exxdiv=None).density_fit()
        ehf  = mf.kernel()
        myadc  = mol_adc.ADC(mf)
        e_mp, t1, t2 = myadc.kernel_gs()
        e_mp = e_mp/(numpy.prod(nmp))

        self.assertAlmostEqual(e_mp, -0.02095766698, 4)

    def test_check_periodic_mp2(self):
        kpts = cell.make_kpts((nmp))
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
        myadc  = adc.KRADC(kmf)
        e_mp, t1, t2 = myadc.kernel_gs()

        self.assertAlmostEqual(e_mp, -0.02095766698, 4)

    def test_check_periodic_mp2_2_high_cost(self):
        nmp = [2,2,2]
        kpts = cell.make_kpts((nmp))
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
        myadc  = adc.KRADC(kmf)
        e_mp2_1, t1, t2 = myadc.kernel_gs()

        mp2 = mp.KMP2(kmf)
        e_mp2_2 = mp2.kernel()[0]

        diff_mp2 = e_mp2_2 - e_mp2_1

        self.assertAlmostEqual(diff_mp2, 0.0000000000, 4)

    def test_check_periodic_mp3_skip(self):
        kpts = cell.make_kpts((nmp))
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
        myadc  = adc.KRADC(kmf)
        myadc.method = 'adc(3)'
        e_mp, t1, t2 = myadc.kernel_gs()

        self.assertAlmostEqual(e_mp, -0.0207106109728, 4)

if __name__ == "__main__":
    print("Ground state calculations for helium")
    unittest.main()
