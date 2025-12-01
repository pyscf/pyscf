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
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf, myadc, myadc_fr
    basis = 'cc-pVDZ'
    mol = gto.Mole()
    mol.atom = '''
        C 0.00000000 0.00000000 -1.18953886
        N 0.00000000 0.00000000 1.01938091
         '''
    mol.basis = {'C': basis,
                 'N': basis,}
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.verbose = 0
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    myadc = adc.ADC(mf)
    myadc_fr = adc.ADC(mf,frozen=(1,1))

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
        myadc.approx_trans_moments = True
        myadc.compute_spin_square = True

        e,v,p,x = myadc.kernel(nroots=5)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0789393239, 6)
        self.assertAlmostEqual(e[1],0.0789393239, 6)
        self.assertAlmostEqual(e[2],0.1397085217, 6)
        self.assertAlmostEqual(e[3],0.2552893678, 6)

        self.assertAlmostEqual(p[0],0.00338362, 6)
        self.assertAlmostEqual(p[1],0.00338362, 6)
        self.assertAlmostEqual(p[2],0.01188668, 6)
        self.assertAlmostEqual(p[3],0.00598859, 6)

        self.assertAlmostEqual(spin[0],0.88753961, 5)
        self.assertAlmostEqual(spin[1],0.88753961, 5)
        self.assertAlmostEqual(spin[2],1.10172022, 5)
        self.assertAlmostEqual(spin[3],2.66684999, 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.8847348053780, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.8847348053781, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 41.0848233538361, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.7558214379312, 6)

    def test_ea_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ea"
        myadc.approx_trans_moments = True

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], -0.1123860177, 6)
        self.assertAlmostEqual(e[1],  0.1309866865, 6)
        self.assertAlmostEqual(e[2],  0.1309866865, 6)
        self.assertAlmostEqual(e[3],  0.1652852298, 6)

        self.assertAlmostEqual(p[0], 0.91914237, 6)
        self.assertAlmostEqual(p[1], 0.92740352, 6)
        self.assertAlmostEqual(p[2], 0.92740352, 6)
        self.assertAlmostEqual(p[3], 0.93979624, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 48.40527094611385, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 47.83221346654948, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 47.83221346654946, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 48.47745904848311, 6)

    def test_ip_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        myadc.approx_trans_moments = True

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4963840022, 6)
        self.assertAlmostEqual(e[1], 0.4963840022, 6)
        self.assertAlmostEqual(e[2], 0.5237162997, 6)
        self.assertAlmostEqual(e[3], 0.5237162997, 6)

        self.assertAlmostEqual(p[0], 0.89871663, 6)
        self.assertAlmostEqual(p[1], 0.89871663, 6)
        self.assertAlmostEqual(p[2], 0.93748642, 6)
        self.assertAlmostEqual(p[3], 0.93748642, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 33.41570000706911, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 33.41570000706909, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 33.25789146072936, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 33.25789146072935, 6)

    def test_ee_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.method_type = "ee"
        myadc_fr.approx_trans_moments = True
        myadc_fr.compute_spin_square = True

        e,v,p,x = myadc_fr.kernel(nroots=5)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0789265253865330, 6)
        self.assertAlmostEqual(e[1],0.0789265253865334, 6)
        self.assertAlmostEqual(e[2],0.1398248585903803, 6)
        self.assertAlmostEqual(e[3],0.2553033553135535, 6)

        self.assertAlmostEqual(p[0],0.0033823617034092, 6)
        self.assertAlmostEqual(p[1],0.0033823617034092, 6)
        self.assertAlmostEqual(p[2],0.0118973560460742, 6)
        self.assertAlmostEqual(p[3],0.0059779519202508, 6)

        self.assertAlmostEqual(spin[0],0.8875461145106943, 5)
        self.assertAlmostEqual(spin[1],0.8875461145106965, 5)
        self.assertAlmostEqual(spin[2],1.1017871591273392, 5)
        self.assertAlmostEqual(spin[3],2.6668023038469877, 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.8844433684455, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.8844433684455, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 41.0841611515388, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.7558140643287, 6)

    def test_ea_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.method_type = "ea"
        myadc_fr.approx_trans_moments = True

        e,v,p,x = myadc_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0], -0.112374942397122, 6)
        self.assertAlmostEqual(e[1],  0.131058880478625, 6)
        self.assertAlmostEqual(e[2],  0.131058880478626, 6)
        self.assertAlmostEqual(e[3],  0.165291577248025, 6)

        self.assertAlmostEqual(p[0], 0.919146532081888, 6)
        self.assertAlmostEqual(p[1], 0.927421334087213, 6)
        self.assertAlmostEqual(p[2], 0.927421334087215, 6)
        self.assertAlmostEqual(p[3], 0.939804864487713, 6)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 48.40483871140096, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 47.83240727171252, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 47.83240727171250, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 48.47714877800658, 6)

    def test_ip_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.method_type = "ip"
        myadc_fr.approx_trans_moments = True

        e,v,p,x = myadc_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4963773005814748, 6)
        self.assertAlmostEqual(e[1], 0.4963773005814786, 6)
        self.assertAlmostEqual(e[2], 0.5237016998217207, 6)
        self.assertAlmostEqual(e[3], 0.5237016998217289, 6)

        self.assertAlmostEqual(p[0], 0.8987400674417594, 6)
        self.assertAlmostEqual(p[1], 0.8987400674417607, 6)
        self.assertAlmostEqual(p[2], 0.9374966427361372, 6)
        self.assertAlmostEqual(p[3], 0.9374966427361376, 6)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 33.41575138552596, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 33.41575138552597, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 33.25804441357201, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 33.25804441357199, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
