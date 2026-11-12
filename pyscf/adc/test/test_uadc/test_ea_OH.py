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
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ea import get_spin_square

def setUpModule():
    global mol, mf, myadc, myadc_fr
    r = 0.969286393
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0.    , -r/2   )],
        ['H', (0., 0.    ,  r/2)],]
    mol.basis = {'O':'aug-cc-pvdz',
                 'H':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.symmetry = False
    mol.spin  = 1
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc_fr = adc.ADC(mf,frozen=(1,1))
    myadc_fr.conv_tol = 1e-12
    myadc_fr.tol_residual = 1e-6

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

    def test_ea_adc2(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387806, 6)

        self.assertAlmostEqual(e[0], -0.048666915263496924, 6)
        self.assertAlmostEqual(e[1], 0.030845983085818485, 6)
        self.assertAlmostEqual(e[2], 0.03253522816723711, 6)

        self.assertAlmostEqual(p[0], 0.9228959646746451, 6)
        self.assertAlmostEqual(p[1], 0.9953765999165347, 6)
        self.assertAlmostEqual(p[2], 0.9956169835481459, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 28.278244381670397, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 58.81399703079512, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.56279801895423, 6)
        self.assertAlmostEqual(spin[0],0.035471582740206564 , 5)
        self.assertAlmostEqual(spin[1],2.0025286730795284 , 5)
        self.assertAlmostEqual(spin[2],1.0073771148805628 , 5)

    def test_ea_adc2_oneroot(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=1)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0], 0.030845983085818485, 6)

        self.assertAlmostEqual(p[0], 0.995376599386389, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 58.81399729877423, 6)
        self.assertAlmostEqual(spin[0],2.0025286731371135 , 5)

    def test_ea_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"

        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0], -0.07750642898162931, 6)
        self.assertAlmostEqual(e[1], 0.029292010466571882, 6)
        self.assertAlmostEqual(e[2], 0.030814773752482663, 6)

        self.assertAlmostEqual(p[0], 0.8323987058794676, 6)
        self.assertAlmostEqual(p[1], 0.9918705979602267, 6)
        self.assertAlmostEqual(p[2], 0.9772861110203499, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 30.37653259884763, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 58.42543220181003, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.033255450114076, 6)
        self.assertAlmostEqual(spin[0],0.003093469410488048 , 5)
        self.assertAlmostEqual(spin[1],2.0026529991584407 , 5)
        self.assertAlmostEqual(spin[2],1.2448957902782012 , 5)


    def test_ea_adc3(self):

        myadc.method = "adc(3)"
        myadc.compute_spin_square = True
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072136, 6)
        self.assertAlmostEqual(myadc.gs_spin_square, 0.7507504248295347, 6)

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]
        myadc.analyze()

        self.assertAlmostEqual(e[0], -0.045097652872531736, 6)
        self.assertAlmostEqual(e[1], 0.03004291636971322, 6)
        self.assertAlmostEqual(e[2], 0.03153897437644345, 6)

        self.assertAlmostEqual(p[0], 0.8722483551941809, 6)
        self.assertAlmostEqual(p[1], 0.992715669957421, 6)
        self.assertAlmostEqual(p[2], 0.9766406731650141, 6)

        self.assertAlmostEqual(spin[0],0.00769901 , 5)
        self.assertAlmostEqual(spin[1],2.00091925 , 5)
        self.assertAlmostEqual(spin[2],1.25284951 , 5)
        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 30.73130805242398, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 58.862530175543064, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.44902995957193, 6)

    def test_ea_adc2_frozen2_spin(self):
        myadc2_fr = adc.ADC(mf, frozen=(2,2))
        myadc2_fr.conv_tol = 1e-12
        myadc2_fr.tol_residual = 1e-6
        myadc2_fr.method_type = "ea"
        e, v, p, x = myadc2_fr.kernel(nroots=3)
        spin = get_spin_square(myadc2_fr._adc_es)[0]
        self.assertAlmostEqual(spin[0], 0.046504830405382336, 5)
        self.assertAlmostEqual(spin[1], 2.005372887404696, 5)
        self.assertAlmostEqual(spin[2], 1.010303633070638, 5)

    def test_ea_adc3_frozen(self):

        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.1741689078419142, 6)

        myadc_fr.method_type = "ea"
        e,v,p,x = myadc_fr.kernel(nroots=3)
        spin = get_spin_square(myadc_fr._adc_es)[0]
        myadc_fr.analyze()

        self.assertAlmostEqual(e[0], -0.04496243643090245, 6)
        self.assertAlmostEqual(e[1],  0.0300414467534960, 6)
        self.assertAlmostEqual(e[2],  0.0315399172709854, 6)

        self.assertAlmostEqual(p[0], 0.8722200055138173, 6)
        self.assertAlmostEqual(p[1], 0.9927152216001243, 6)
        self.assertAlmostEqual(p[2], 0.9767546870671993, 6)

        self.assertAlmostEqual(spin[0],0.00773845 , 5)
        self.assertAlmostEqual(spin[1],2.00092271 , 5)
        self.assertAlmostEqual(spin[2],1.25198071 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 30.733555576389282, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 58.860839887579836, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.44849342907379, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods for open-shell molecule")
    unittest.main()
