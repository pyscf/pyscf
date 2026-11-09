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

    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    myadc = adc.ADC(mf)
    myadc_fr = adc.ADC(mf,frozen=[1,1])

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', myadc.mo_coeff_hf[0], dm_a, myadc.mo_coeff_hf[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', myadc.mo_coeff_hf[1], dm_b, myadc.mo_coeff_hf[1].conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao_a+dm_ao_b)
    return r2

def rdms_test_fr(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', myadc_fr.mo_coeff_hf[0], dm_a, myadc_fr.mo_coeff_hf[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', myadc_fr.mo_coeff_hf[1], dm_b, myadc_fr.mo_coeff_hf[1].conj())
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

        self.assertAlmostEqual(e[0],0.0540102949, 6)
        self.assertAlmostEqual(e[1],0.0540102949, 6)
        self.assertAlmostEqual(e[2],0.0955821322, 6)
        self.assertAlmostEqual(e[3],0.2557350668, 6)

        self.assertAlmostEqual(p[0],0.003520434343753506, 6)
        self.assertAlmostEqual(p[1],0.0035204343437535376, 6)
        self.assertAlmostEqual(p[2],0.017249528706218778, 6)
        self.assertAlmostEqual(p[3],0.001395770972926244, 6)

        self.assertAlmostEqual(spin[0],0.7556349446790636 , 5)
        self.assertAlmostEqual(spin[1],0.7556349446790618 , 5)
        self.assertAlmostEqual(spin[2],0.7679986999933224 , 5)
        self.assertAlmostEqual(spin[3],2.838185371306401 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.55628130430309, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.55628130430308, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.851564937311785, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.681304383364036, 4)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0164020499, 6)
        self.assertAlmostEqual(e[1],0.0164020499, 6)
        self.assertAlmostEqual(e[2],0.0530368555, 6)
        self.assertAlmostEqual(e[3],0.1763402895, 6)

        self.assertAlmostEqual(p[0],0.0009675742204159537, 6)
        self.assertAlmostEqual(p[1],0.0009675742204159729, 6)
        self.assertAlmostEqual(p[2],0.008851319604914407, 6)
        self.assertAlmostEqual(p[3],0.0005039488068446622, 6)

        self.assertAlmostEqual(spin[0],0.7556233004604636 , 5)
        self.assertAlmostEqual(spin[1],0.7556233004604627 , 5)
        self.assertAlmostEqual(spin[2],0.7620856932087925 , 5)
        self.assertAlmostEqual(spin[3],3.3067380865227256 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.264031374751276, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.26403137475129, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.56105752654338, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.69667140447979, 4)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362916619, 6)
        self.assertAlmostEqual(e[1],0.0362916619, 6)
        self.assertAlmostEqual(e[2],0.1200404011, 6)
        self.assertAlmostEqual(e[3],0.1747467421, 6)

        self.assertAlmostEqual(p[0],0.002155621030536797, 6)
        self.assertAlmostEqual(p[1],0.002155621030536799, 6)
        self.assertAlmostEqual(p[2],0.020709795512996683, 6)
        self.assertAlmostEqual(p[3],0.0014228575270046567, 6)

        self.assertAlmostEqual(spin[0],0.75368690 , 5)
        self.assertAlmostEqual(spin[1],0.75368690 , 5)
        self.assertAlmostEqual(spin[2],0.79336635 , 5)
        self.assertAlmostEqual(spin[3],3.44302505 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.26673509617428, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.26673509617426, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.55540890898248, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.63765650861586, 4)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.max_memory = 20
        myadc_fr.incore_complete = False

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362650906, 6)
        self.assertAlmostEqual(e[1],0.0362650906, 6)
        self.assertAlmostEqual(e[2],0.1200928094, 6)
        self.assertAlmostEqual(e[3],0.1747679417, 6)

        self.assertAlmostEqual(p[0],0.0021549082418727923, 6)
        self.assertAlmostEqual(p[1],0.0021549082418728196, 6)
        self.assertAlmostEqual(p[2],0.020717608821126277, 6)
        self.assertAlmostEqual(p[3],0.00142498366161091, 6)

        self.assertAlmostEqual(spin[0],0.75368608 , 5)
        self.assertAlmostEqual(spin[1],0.75368608 , 5)
        self.assertAlmostEqual(spin[2],0.79344127 , 5)
        self.assertAlmostEqual(spin[3],3.44307487 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][0],dm1_exc[1][0]), 41.26600829806355, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][1],dm1_exc[1][1]), 41.266008298063525, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][2],dm1_exc[1][2]), 40.554816114525906, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][3],dm1_exc[1][3]), 40.63735349632753, 4)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
