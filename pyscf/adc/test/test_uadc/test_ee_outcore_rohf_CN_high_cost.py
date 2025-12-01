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

        self.assertAlmostEqual(p[0],0.00330109, 6)
        self.assertAlmostEqual(p[1],0.00330109, 6)
        self.assertAlmostEqual(p[2],0.03417973, 6)
        self.assertAlmostEqual(p[3],0.00247641, 6)

        self.assertAlmostEqual(spin[0],0.75455998 , 5)
        self.assertAlmostEqual(spin[1],0.75455998 , 5)
        self.assertAlmostEqual(spin[2],0.76652057 , 5)
        self.assertAlmostEqual(spin[3],2.83811703 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.35496697903834, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.35496697903835, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.52967870024318, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.49642339005783, 4)

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

        self.assertAlmostEqual(p[0],0.00089364, 6)
        self.assertAlmostEqual(p[1],0.00089364, 6)
        self.assertAlmostEqual(p[2],0.01674309, 6)
        self.assertAlmostEqual(p[3],0.00080394, 6)

        self.assertAlmostEqual(spin[0],0.75474307 , 5)
        self.assertAlmostEqual(spin[1],0.75474307 , 5)
        self.assertAlmostEqual(spin[2],0.76071932 , 5)
        self.assertAlmostEqual(spin[3],3.30672580 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.07764719625411, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.07764719625408, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.27044834802643, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.64183214575419, 4)

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

        self.assertAlmostEqual(p[0],0.00199181, 6)
        self.assertAlmostEqual(p[1],0.00199181, 6)
        self.assertAlmostEqual(p[2],0.02121601, 6)
        self.assertAlmostEqual(p[3],0.00134579, 6)

        self.assertAlmostEqual(spin[0],0.75778870 , 5)
        self.assertAlmostEqual(spin[1],0.75778870 , 5)
        self.assertAlmostEqual(spin[2],0.79970595 , 5)
        self.assertAlmostEqual(spin[3],3.45250153 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.36517719120032, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.36517719120031, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.62550586260885, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.68911000506814, 4)

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

        self.assertAlmostEqual(p[0],0.00199151, 6)
        self.assertAlmostEqual(p[1],0.00199151, 6)
        self.assertAlmostEqual(p[2],0.02123242, 6)
        self.assertAlmostEqual(p[3],0.00134849, 6)

        self.assertAlmostEqual(spin[0],0.75778631 , 5)
        self.assertAlmostEqual(spin[1],0.75778631 , 5)
        self.assertAlmostEqual(spin[2],0.79978552 , 5)
        self.assertAlmostEqual(spin[3],3.45255970 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][0],dm1_exc[1][0]), 41.3645155293106, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][1],dm1_exc[1][1]), 41.3645155293106, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][2],dm1_exc[1][2]), 40.6249921731014, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][3],dm1_exc[1][3]), 40.6888648482933, 4)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
