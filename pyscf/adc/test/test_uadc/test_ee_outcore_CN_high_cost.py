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
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf, myadc, myadc_fr
    basis = 'cc-pVDZ'
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
        C 0.00000000 0.00000000 -1.18953886
        N 0.00000000 0.00000000 1.01938091
         '''
    mol.basis = {'C': basis,
                 'N': basis,}
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    myadc = adc.ADC(mf)
    myadc_fr = adc.ADC(mf,frozen=[1,1])

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', mf.mo_coeff[0], dm_a, mf.mo_coeff[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', mf.mo_coeff[1], dm_b, mf.mo_coeff[1].conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao_a+dm_ao_b)
    return r2

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0789393239, 6)
        self.assertAlmostEqual(e[1],0.0789393239, 6)
        self.assertAlmostEqual(e[2],0.1397085217, 6)
        self.assertAlmostEqual(e[3],0.2552893678, 6)

        self.assertAlmostEqual(p[0],0.00403599, 6)
        self.assertAlmostEqual(p[1],0.00403599, 6)
        self.assertAlmostEqual(p[2],0.02229693, 6)
        self.assertAlmostEqual(p[3],0.00597127, 6)

        self.assertAlmostEqual(spin[0],0.81903415 , 5)
        self.assertAlmostEqual(spin[1],0.81903415 , 5)
        self.assertAlmostEqual(spin[2],0.97833065 , 5)
        self.assertAlmostEqual(spin[3],2.70435538 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.645076886712, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.645076886713, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.824586142278, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.623139654484, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0066160563, 6)
        self.assertAlmostEqual(e[1],0.0066160563, 6)
        self.assertAlmostEqual(e[2],0.0674217414, 6)
        self.assertAlmostEqual(e[3],0.1755586417, 6)

        self.assertAlmostEqual(p[0],0.00027751, 6)
        self.assertAlmostEqual(p[1],0.00027751, 6)
        self.assertAlmostEqual(p[2],0.01004905, 6)
        self.assertAlmostEqual(p[3],0.00001243, 6)

        self.assertAlmostEqual(spin[0],0.76591155 , 5)
        self.assertAlmostEqual(spin[1],0.76591155 , 5)
        self.assertAlmostEqual(spin[2],0.77023553 , 5)
        self.assertAlmostEqual(spin[3],4.03885373 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.1498995698491, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.1498995698490, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.3891114650266, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.5160045387122, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0432157465, 6)
        self.assertAlmostEqual(e[1],0.0432157465, 6)
        self.assertAlmostEqual(e[2],0.1276752421, 6)
        self.assertAlmostEqual(e[3],0.1848576902, 6)

        self.assertAlmostEqual(p[0],0.00208411, 6)
        self.assertAlmostEqual(p[1],0.00208411, 6)
        self.assertAlmostEqual(p[2],0.01526701, 6)
        self.assertAlmostEqual(p[3],0.00009135, 6)

        self.assertAlmostEqual(spin[0],0.78024113 , 5)
        self.assertAlmostEqual(spin[1],0.78024113 , 5)
        self.assertAlmostEqual(spin[2],0.79826249 , 5)
        self.assertAlmostEqual(spin[3],4.10509903 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.3590831008717, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.3590831008716, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.6275073222460, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.6750649892015, 6)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.max_memory = 20
        myadc_fr.incore_complete = False

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.04319092877771163, 6)
        self.assertAlmostEqual(e[1],0.04319092877771207, 6)
        self.assertAlmostEqual(e[2],0.12770468024295392, 6)
        self.assertAlmostEqual(e[3],0.18489326630403885, 6)

        self.assertAlmostEqual(p[0],0.00208418, 6)
        self.assertAlmostEqual(p[1],0.00208418, 6)
        self.assertAlmostEqual(p[2],0.01528568, 6)
        self.assertAlmostEqual(p[3],0.00009021, 6)

        self.assertAlmostEqual(spin[0],0.78027442, 5)
        self.assertAlmostEqual(spin[1],0.78027442, 5)
        self.assertAlmostEqual(spin[2],0.79839650, 5)
        self.assertAlmostEqual(spin[3],4.10501620, 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.35867463912157, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.35867463912155, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.62700757491586, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.67487379862491, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
