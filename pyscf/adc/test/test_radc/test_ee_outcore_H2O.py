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
# Author: Terrence Stahl <terrencestahl1@@gmail.com>
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc, myadc_fr
    mol = gto.Mole()
    r = 0.957492
    x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
    y = r * math.cos(104.468205* math.pi/(2 * 180.0))
    mol.atom = [
        ['O', (0., 0.    , 0)],
        ['H', (0., -x, y)],
        ['H', (0., x , y)],]
    mol.basis = {'H': 'cc-pVDZ',
                 'O': 'cc-pVDZ',}
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc_fr = adc.ADC(mf,frozen=1)

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(p[0], 0.02774680081092312, 6)
        self.assertAlmostEqual(p[1], 8.90646730745011e-29, 6)
        self.assertAlmostEqual(p[2], 0.09770117474911057, 6)
        self.assertAlmostEqual(p[3], 0.07375673165508137, 6)

        self.assertAlmostEqual(e[0],0.2971167095 , 6)
        self.assertAlmostEqual(e[1],0.3724791374 , 6)
        self.assertAlmostEqual(e[2],0.3935563988 , 6)
        self.assertAlmostEqual(e[3],0.4709279042 , 6)


    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(p[0], 0.0254619534304077, 6)
        self.assertAlmostEqual(p[1], 5.067710484943722e-29, 6)
        self.assertAlmostEqual(p[2], 0.0917847064014669, 6)
        self.assertAlmostEqual(p[3], 0.0674078023930496, 6)

        self.assertAlmostEqual(e[0],0.2794713515, 6)
        self.assertAlmostEqual(e[1],0.3563942404, 6)
        self.assertAlmostEqual(e[2],0.3757585048, 6)
        self.assertAlmostEqual(e[3],0.4551913585, 6)


    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2107769014592799, 6)

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3053164039, 6)
        self.assertAlmostEqual(e[1],0.3790532845, 6)
        self.assertAlmostEqual(e[2],0.4019531805, 6)
        self.assertAlmostEqual(e[3],0.4772033490, 6)

        self.assertAlmostEqual(p[0], 0.02702943, 6)
        self.assertAlmostEqual(p[1], 0.00000000, 6)
        self.assertAlmostEqual(p[2], 0.09696533, 6)
        self.assertAlmostEqual(p[3], 0.07673359, 6)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ee"
        myadc_fr.max_memory = 20
        myadc_fr.incore_complete = False
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.20864693991051741, 6)

        myadcee_fr = adc.radc_ee.RADCEE(myadc_fr)
        e,v,p,x = myadcee_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3052262994945224, 6)
        self.assertAlmostEqual(e[1],0.3789606827167341, 6)
        self.assertAlmostEqual(e[2],0.4018990744972834, 6)
        self.assertAlmostEqual(e[3],0.4771607225277996, 6)

        self.assertAlmostEqual(p[0], 0.02702067, 6)
        self.assertAlmostEqual(p[1], 0.00000000, 6)
        self.assertAlmostEqual(p[2], 0.09699817, 6)
        self.assertAlmostEqual(p[3], 0.07684783, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for water molecule")
    unittest.main()
