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
    mol.verbose = 0
    mol.symmetry = "c2v"
    mol.spin = 1
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
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0789393239, 6)
        self.assertAlmostEqual(e[1],0.0789393239, 6)
        self.assertAlmostEqual(e[2],0.1397085217, 6)
        self.assertAlmostEqual(e[3],0.2552893678, 6)

        self.assertAlmostEqual(p[0],0.0033836167048380095, 6)
        self.assertAlmostEqual(p[1],0.0033836167048380754, 6)
        self.assertAlmostEqual(p[2],0.011886684786878744, 6)
        self.assertAlmostEqual(p[3],0.005988588887973597, 6)

        self.assertAlmostEqual(spin[0],0.8875396093786074 , 5)
        self.assertAlmostEqual(spin[1],0.8875396093786092 , 5)
        self.assertAlmostEqual(spin[2],1.1017202202637657 , 5)
        self.assertAlmostEqual(spin[3],2.6668499895288047 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.88473480537788, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.88473480537791, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 41.084823353836015, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.755821437931054, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0066160563, 6)
        self.assertAlmostEqual(e[1],0.0066160563, 6)
        self.assertAlmostEqual(e[2],0.0674217414, 6)
        self.assertAlmostEqual(e[3],0.1755586417, 6)

        self.assertAlmostEqual(p[0],0.00023276121952183984, 6)
        self.assertAlmostEqual(p[1],0.00023276121952189974, 6)
        self.assertAlmostEqual(p[2],0.005510428569707829, 6)
        self.assertAlmostEqual(p[3],0.0001268377803786541, 6)

        self.assertAlmostEqual(spin[0],0.8237305796467203 , 5)
        self.assertAlmostEqual(spin[1],0.8237305796467247 , 5)
        self.assertAlmostEqual(spin[2],0.8778829114629003 , 5)
        self.assertAlmostEqual(spin[3],4.015400829318011 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.35233574668308, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.352335746683096, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.61156613734314, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.58559677744399, 6)

    def test_ee_adc2x_cis(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4, guess="cis")
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0066160562998973, 6)
        self.assertAlmostEqual(e[1],0.0066160562998976, 6)
        self.assertAlmostEqual(e[2],0.0674217413926688, 6)
        self.assertAlmostEqual(e[3],0.1755586417565505, 6)

        self.assertAlmostEqual(p[0],0.0002327636933497959, 6)
        self.assertAlmostEqual(p[1],0.00023276369334981938, 6)
        self.assertAlmostEqual(p[2],0.005510407283637052, 6)
        self.assertAlmostEqual(p[3],0.00012684377133060276, 6)

        self.assertAlmostEqual(spin[0],0.8237303141950063 , 5)
        self.assertAlmostEqual(spin[1],0.8237303141950072 , 5)
        self.assertAlmostEqual(spin[2],0.8778834311955066 , 5)
        self.assertAlmostEqual(spin[3],4.015398410655937 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.352338245254955, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.35233824525502, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.61156439086489, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.58560599435299, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0432157465, 6)
        self.assertAlmostEqual(e[1],0.0432157465, 6)
        self.assertAlmostEqual(e[2],0.1276752421, 6)
        self.assertAlmostEqual(e[3],0.1848576902, 6)

        self.assertAlmostEqual(p[0],0.001929265751970028, 6)
        self.assertAlmostEqual(p[1],0.0019292657519700766, 6)
        self.assertAlmostEqual(p[2],0.012786981363066094, 6)
        self.assertAlmostEqual(p[3],0.00014257794859825253, 6)

        self.assertAlmostEqual(spin[0],0.80195386, 5)
        self.assertAlmostEqual(spin[1],0.80195386, 5)
        self.assertAlmostEqual(spin[2],0.82700993, 5)
        self.assertAlmostEqual(spin[3],4.08789116, 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.33932648981075, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.33932648981071, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.602868264703275, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.6237151238264, 6)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0431909287777104, 6)
        self.assertAlmostEqual(e[1],0.0431909287777112, 6)
        self.assertAlmostEqual(e[2],0.1277046802429511, 6)
        self.assertAlmostEqual(e[3],0.1848932663040389, 6)

        self.assertAlmostEqual(p[0],0.001928948590435291, 6)
        self.assertAlmostEqual(p[1],0.0019289485904353475, 6)
        self.assertAlmostEqual(p[2],0.012796188903623412, 6)
        self.assertAlmostEqual(p[3],0.00014137665143170943, 6)

        self.assertAlmostEqual(spin[0],0.80197118, 5)
        self.assertAlmostEqual(spin[1],0.80197118, 5)
        self.assertAlmostEqual(spin[2],0.82709336, 5)
        self.assertAlmostEqual(spin[3],4.08781662, 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.33868050571835, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.33868050571836, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.602128264908835, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.623400412934004, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
