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

        self.assertAlmostEqual(p[0],0.00403804, 6)
        self.assertAlmostEqual(p[1],0.00403804, 6)
        self.assertAlmostEqual(p[2],0.02230072, 6)
        self.assertAlmostEqual(p[3],0.00596792, 6)

        self.assertAlmostEqual(spin[0],0.81897586, 5)
        self.assertAlmostEqual(spin[1],0.81897586, 5)
        self.assertAlmostEqual(spin[2],0.97818035, 5)
        self.assertAlmostEqual(spin[3],2.70419098, 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.6441006032869, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.6441006032869, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.8235523484000, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.6217002530918, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0065704302, 6)
        self.assertAlmostEqual(e[1],0.0065704302, 6)
        self.assertAlmostEqual(e[2],0.0673712996, 6)
        self.assertAlmostEqual(e[3],0.1755822503, 6)

        self.assertAlmostEqual(p[0],0.00027567, 6)
        self.assertAlmostEqual(p[1],0.00027567 , 6)
        self.assertAlmostEqual(p[2],0.01004173 , 6)
        self.assertAlmostEqual(p[3],0.00001197 , 6)

        self.assertAlmostEqual(spin[0],0.76587776 , 5)
        self.assertAlmostEqual(spin[1],0.76587776 , 5)
        self.assertAlmostEqual(spin[2],0.77014916 , 5)
        self.assertAlmostEqual(spin[3],4.03860924 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.14930526794730, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.14930526794730, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.38829604487761, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.51548534697557, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0431409359, 6)
        self.assertAlmostEqual(e[1],0.0431409359, 6)
        self.assertAlmostEqual(e[2],0.1276592929, 6)
        self.assertAlmostEqual(e[3],0.1848566262, 6)

        self.assertAlmostEqual(p[0],0.00208140, 6)
        self.assertAlmostEqual(p[1],0.00208140 , 6)
        self.assertAlmostEqual(p[2],0.01526421 , 6)
        self.assertAlmostEqual(p[3],0.00008967 , 6)

        self.assertAlmostEqual(spin[0],0.78026235 , 5)
        self.assertAlmostEqual(spin[1],0.78026235 , 5)
        self.assertAlmostEqual(spin[2],0.79814541 , 5)
        self.assertAlmostEqual(spin[3],4.10490908 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.35853494368054, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.35853494368057, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.62687509720155, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.67459269393425, 6)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0431161470227286, 6)
        self.assertAlmostEqual(e[1],0.0431161470227291, 6)
        self.assertAlmostEqual(e[2],0.1276887701715607, 6)
        self.assertAlmostEqual(e[3],0.1848923370493608, 6)

        self.assertAlmostEqual(p[0],0.00208146, 6)
        self.assertAlmostEqual(p[1],0.00208146 , 6)
        self.assertAlmostEqual(p[2],0.01528284 , 6)
        self.assertAlmostEqual(p[3],0.00008856 , 6)

        self.assertAlmostEqual(spin[0],0.78029468 , 5)
        self.assertAlmostEqual(spin[1],0.78029468 , 5)
        self.assertAlmostEqual(spin[2],0.79827929 , 5)
        self.assertAlmostEqual(spin[3],4.10483825 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.35811760272720, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.35811760272718, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.62637555265807, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.67439797320911, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
