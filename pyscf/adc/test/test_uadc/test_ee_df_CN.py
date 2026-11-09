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

    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
    myadc_fr = adc.ADC(mf,frozen=(1,1)).density_fit('cc-pvdz-ri')

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

        self.assertAlmostEqual(e[0],0.0789805434, 6)
        self.assertAlmostEqual(e[1],0.0789805434, 6)
        self.assertAlmostEqual(e[2],0.1397261293, 6)
        self.assertAlmostEqual(e[3],0.2553471934, 6)

        self.assertAlmostEqual(p[0],0.0033861068673907346, 6)
        self.assertAlmostEqual(p[1],0.0033861068673907667, 6)
        self.assertAlmostEqual(p[2],0.011890563461095568, 6)
        self.assertAlmostEqual(p[3],0.005985489379261925, 6)

        self.assertAlmostEqual(spin[0],0.8875144316618577, 5)
        self.assertAlmostEqual(spin[1],0.8875144316618604, 5)
        self.assertAlmostEqual(spin[2],1.1016310897029804, 5)
        self.assertAlmostEqual(spin[3],2.6666933152976453, 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.88467329384436, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.88467329384434, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 41.084702666016916, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.75530929574556, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0065704302, 6)
        self.assertAlmostEqual(e[1],0.0065704302, 6)
        self.assertAlmostEqual(e[2],0.0673712996, 6)
        self.assertAlmostEqual(e[3],0.1755822503, 6)

        self.assertAlmostEqual(p[0],0.0002312493777703869, 6)
        self.assertAlmostEqual(p[1],0.00023124937777039835 , 6)
        self.assertAlmostEqual(p[2],0.005507012979274538 , 6)
        self.assertAlmostEqual(p[3],0.00012551576193663185 , 6)

        self.assertAlmostEqual(spin[0],0.8237201452514453 , 5)
        self.assertAlmostEqual(spin[1],0.8237201452514382 , 5)
        self.assertAlmostEqual(spin[2],0.8778441345187362 , 5)
        self.assertAlmostEqual(spin[3],4.015184014464251 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.35250244261326, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.352502442613286, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.6115145611751, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.58549688684841, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0431409359, 6)
        self.assertAlmostEqual(e[1],0.0431409359, 6)
        self.assertAlmostEqual(e[2],0.1276592929, 6)
        self.assertAlmostEqual(e[3],0.1848566262, 6)

        self.assertAlmostEqual(p[0],0.0019263926628261278, 6)
        self.assertAlmostEqual(p[1],0.0019263926628261664 , 6)
        self.assertAlmostEqual(p[2],0.012784203422431377 , 6)
        self.assertAlmostEqual(p[3],0.00014073448302803367 , 6)

        self.assertAlmostEqual(spin[0],0.80196836 , 5)
        self.assertAlmostEqual(spin[1],0.80196836 , 5)
        self.assertAlmostEqual(spin[2],0.82690040 , 5)
        self.assertAlmostEqual(spin[3],4.08773668 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.33890768604357, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.338907686043626, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.6023087504581, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.62329970544882, 6)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0431161470227286, 6)
        self.assertAlmostEqual(e[1],0.0431161470227291, 6)
        self.assertAlmostEqual(e[2],0.1276887701715607, 6)
        self.assertAlmostEqual(e[3],0.1848923370493608, 6)

        self.assertAlmostEqual(p[0],0.0019260668350830608, 6)
        self.assertAlmostEqual(p[1],0.0019260668350831122 , 6)
        self.assertAlmostEqual(p[2],0.012793384294932151 , 6)
        self.assertAlmostEqual(p[3],0.00013955449595714876 , 6)

        self.assertAlmostEqual(spin[0],0.80198494 , 5)
        self.assertAlmostEqual(spin[1],0.80198494 , 5)
        self.assertAlmostEqual(spin[2],0.82698365 , 5)
        self.assertAlmostEqual(spin[3],4.08766652 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.33826198312952, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.33826198312958, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.60156863941078, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.62298601854761, 6)

    def test_ee_adc2_naf(self):
        myadc_naf = adc.ADC(mf).density_fit('cc-pvdz-ri')
        myadc_naf.method = "adc(2)"
        myadc_naf.method_type = "ee"
        myadc_naf.if_naf = True
        myadc_naf.thresh_naf = 1e-3
        e,v,p,x = myadc_naf.kernel(nroots=4)
        spin = get_spin_square(myadc_naf._adc_es)[0]

        self.assertAlmostEqual(e[0],0.07898013289167467, 6)
        self.assertAlmostEqual(e[1],0.0789801328916748, 6)
        self.assertAlmostEqual(e[2],0.13972276465255035, 6)
        self.assertAlmostEqual(e[3],0.25534785268385707, 6)

        self.assertAlmostEqual(p[0],0.003386087665166727, 6)
        self.assertAlmostEqual(p[1],0.0033860876651667276, 6)
        self.assertAlmostEqual(p[2],0.011890199395458148, 6)
        self.assertAlmostEqual(p[3],0.005986552520652376, 6)

        self.assertAlmostEqual(spin[0],0.8875164202744701, 5)
        self.assertAlmostEqual(spin[1],0.8875164202744665, 5)
        self.assertAlmostEqual(spin[2],1.1016265497526874, 5)
        self.assertAlmostEqual(spin[3],2.6666909740781417, 5)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
