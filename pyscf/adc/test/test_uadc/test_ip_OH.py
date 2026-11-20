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
from pyscf.adc.uadc_ip import get_spin_square

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

    def test_ip_adc2(self):

        myadc.compute_spin_square = True
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(myadc.gs_spin_square, 0.7525425955019274, 6)

        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387906, 6)

        self.assertAlmostEqual(e[0], 0.4342864327917968, 6)
        self.assertAlmostEqual(e[1], 0.47343844767816784, 6)
        self.assertAlmostEqual(e[2], 0.5805631452815511, 6)

        self.assertAlmostEqual(p[0], 0.9058488149525854, 6)
        self.assertAlmostEqual(p[1], 0.8979216632613441, 6)
        self.assertAlmostEqual(p[2], 0.9115851423147068, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.15775355276691, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.23232918515558, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 15.88664556512575, 6)
        self.assertAlmostEqual(spin[0],2.001826471842401 , 5)
        self.assertAlmostEqual(spin[1],1.085055193111033 , 5)
        self.assertAlmostEqual(spin[2],2.001630820669052 , 5)

    def test_ip_adc2x(self):

        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0], 0.4389083582117278, 6)
        self.assertAlmostEqual(e[1], 0.45720829251439343, 6)
        self.assertAlmostEqual(e[2], 0.5588942056812034, 6)

        self.assertAlmostEqual(p[0], 0.9161237660123829, 6)
        self.assertAlmostEqual(p[1], 0.6988041757561605, 6)
        self.assertAlmostEqual(p[2], 0.21286126206804035, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.253367298060837, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.55865056530131, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 17.194647427238838, 6)
        self.assertAlmostEqual(spin[0],2.001538757123655 , 5)
        self.assertAlmostEqual(spin[1],1.8256843856500167 , 5)
        self.assertAlmostEqual(spin[2],0.1958545372901912 , 5)

    def test_ip_adc3(self):

        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072194, 6)

        myadc.method_type = "ip"
        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]
        myadc.analyze()

        self.assertAlmostEqual(e[0], 0.4794423247368058, 6)
        self.assertAlmostEqual(e[1], 0.4872370596653387, 6)
        self.assertAlmostEqual(e[2], 0.5726961805214643, 6)

        self.assertAlmostEqual(p[0], 0.9285086275683779, 6)
        self.assertAlmostEqual(p[1], 0.5189724901732271, 6)
        self.assertAlmostEqual(p[2], 0.4065986694956914, 6)

        self.assertAlmostEqual(spin[0],2.00025760 , 5)
        self.assertAlmostEqual(spin[1],1.96962920 , 5)
        self.assertAlmostEqual(spin[2],0.05053658 , 5)
        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.470209264118743, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.889353231496766, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 17.095466728139158, 6)

    def test_ip_adc2_frozen2_spin(self):
        myadc2_fr = adc.ADC(mf, frozen=(2,2))
        myadc2_fr.conv_tol = 1e-12
        myadc2_fr.tol_residual = 1e-6
        myadc2_fr.method_type = "ip"
        e, v, p, x = myadc2_fr.kernel(nroots=3)
        spin = get_spin_square(myadc2_fr._adc_es)[0]
        self.assertAlmostEqual(spin[0], 2.0049011496094096, 5)
        self.assertAlmostEqual(spin[1], 1.0910028461235637, 5)
        self.assertAlmostEqual(spin[2], 2.0044862342974943, 5)

    def test_ip_adc3_frozen(self):

        myadc_fr.method = "adc(3)"
        myadc_fr.compute_spin_square = True
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.17416890784191427, 6)
        self.assertAlmostEqual(myadc_fr.gs_spin_square, 0.7507539497201545, 6)

        myadc_fr.method_type = "ip"
        e,v,p,x = myadc_fr.kernel(nroots=3)
        spin = get_spin_square(myadc_fr._adc_es)[0]
        myadc_fr.analyze()

        self.assertAlmostEqual(e[0], 0.479335828293046, 6)
        self.assertAlmostEqual(e[1], 0.487239087759452, 6)
        self.assertAlmostEqual(e[2], 0.572673343025919, 6)

        self.assertAlmostEqual(p[0], 0.9285016651480348, 6)
        self.assertAlmostEqual(p[1], 0.5201406212527856, 6)
        self.assertAlmostEqual(p[2], 0.4054240789596952, 6)

        self.assertAlmostEqual(spin[0],2.00026048 , 5)
        self.assertAlmostEqual(spin[1],1.96931720 , 5)
        self.assertAlmostEqual(spin[2],0.05083684 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.46939396144714, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.887893323133593, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 17.09616779293735, 6)

    def test_ip_adc3_oneroot(self):

        myadc.method = "adc(3)"
        e,v,p,x = myadc.kernel()
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0], 0.4794423247368058, 6)

        self.assertAlmostEqual(p[0], 0.9285086275683778, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.470209264118743, 6)
        self.assertAlmostEqual(spin[0],2.00025760 , 5)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
