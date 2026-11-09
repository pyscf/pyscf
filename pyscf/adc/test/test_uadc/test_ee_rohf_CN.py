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
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0540102949, 6)
        self.assertAlmostEqual(e[1],0.0540102949, 6)
        self.assertAlmostEqual(e[2],0.0955821322, 6)
        self.assertAlmostEqual(e[3],0.2557350669, 6)

        self.assertAlmostEqual(p[0],0.0035204343437534105, 6)
        self.assertAlmostEqual(p[1],0.0035204343437535033, 6)
        self.assertAlmostEqual(p[2],0.01724952870621889, 6)
        self.assertAlmostEqual(p[3],0.001395770972926273, 6)

        self.assertAlmostEqual(spin[0],0.7556349446790627 , 5)
        self.assertAlmostEqual(spin[1],0.7556349446790627 , 5)
        self.assertAlmostEqual(spin[2],0.7679986999933242 , 5)
        self.assertAlmostEqual(spin[3],2.8381853713064 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.55628130430313, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.556281304303084, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.851564937311835, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.68130438336402, 4)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0164020499, 6)
        self.assertAlmostEqual(e[1],0.0164020499, 6)
        self.assertAlmostEqual(e[2],0.0530368555, 6)
        self.assertAlmostEqual(e[3],0.1763402895, 6)

        self.assertAlmostEqual(p[0],0.00096757422041598, 6)
        self.assertAlmostEqual(p[1],0.0009675742204160084, 6)
        self.assertAlmostEqual(p[2],0.008851319604914407, 6)
        self.assertAlmostEqual(p[3],0.000503948806844661, 6)

        self.assertAlmostEqual(spin[0],0.7556233004604609 , 5)
        self.assertAlmostEqual(spin[1],0.7556233004604644 , 5)
        self.assertAlmostEqual(spin[2],0.7620856932087916 , 5)
        self.assertAlmostEqual(spin[3],3.3067380865227145 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.26403137475124, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.264031374751305, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.56105752654335, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.69667140447981, 4)

    def test_ee_adc3_high_cost(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362916619, 6)
        self.assertAlmostEqual(e[1],0.0362916619, 6)
        self.assertAlmostEqual(e[2],0.1200404011, 6)
        self.assertAlmostEqual(e[3],0.1747467421, 6)

        self.assertAlmostEqual(p[0],0.0021556210305367987, 6)
        self.assertAlmostEqual(p[1],0.00215562103053683, 6)
        self.assertAlmostEqual(p[2],0.020709795512996728, 6)
        self.assertAlmostEqual(p[3],0.0014228575270046576, 6)

        self.assertAlmostEqual(spin[0],0.75368690 , 5)
        self.assertAlmostEqual(spin[1],0.75368690 , 5)
        self.assertAlmostEqual(spin[2],0.79336625 , 5)
        self.assertAlmostEqual(spin[3],3.44302560 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.26673509617422, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.266735096174266, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.555408908982464, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.6376565086159, 4)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362650906, 6)
        self.assertAlmostEqual(e[1],0.0362650906, 6)
        self.assertAlmostEqual(e[2],0.1200928094, 6)
        self.assertAlmostEqual(e[3],0.1747679417, 6)

        self.assertAlmostEqual(p[0],0.0021549082418728218, 6)
        self.assertAlmostEqual(p[1],0.002154908241872828, 6)
        self.assertAlmostEqual(p[2],0.020717608821126506, 6)
        self.assertAlmostEqual(p[3],0.001424983661610845, 6)

        self.assertAlmostEqual(spin[0],0.75368608 , 5)
        self.assertAlmostEqual(spin[1],0.75368608 , 5)
        self.assertAlmostEqual(spin[2],0.79344160 , 5)
        self.assertAlmostEqual(spin[3],3.44307440 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][0],dm1_exc[1][0]), 41.26600829806353, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][1],dm1_exc[1][1]), 41.26600829806356, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][2],dm1_exc[1][2]), 40.55481611452593, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][3],dm1_exc[1][3]), 40.63735349632756, 4)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
