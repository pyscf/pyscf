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

def setUpModule():
    global mol, mf, myadc
    mol = gto.Mole()
    mol.atom = [
        ['P', (0., 0.    , 0.)],]
    mol.basis = {'P':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.spin = 3
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.irrep_nelec = {'A1g':(3,3),'E1ux':(2,1),'E1uy':(2,1),'A1u':(2,1)}
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

def rdms_test(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', mf.mo_coeff[0], dm_a, mf.mo_coeff[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', mf.mo_coeff[1], dm_b, mf.mo_coeff[1].conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao_a+dm_ao_b)
    return r2

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        myadc.ncvs = 2
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.073511478250083, 6)

        e,v,p,x = myadc.kernel(nroots=6)

        self.assertAlmostEqual(e[0], 7.29065747318148, 6)
        self.assertAlmostEqual(e[1], 7.31506012579633, 6)
        self.assertAlmostEqual(e[2], 7.92530170069818, 6)
        self.assertAlmostEqual(e[3], 7.92530170069818, 6)
        self.assertAlmostEqual(e[4], 7.92530170069818, 6)
        self.assertAlmostEqual(e[5], 7.92530170069821, 6)

        self.assertAlmostEqual(p[0], 0.86274208688106, 6)
        self.assertAlmostEqual(p[1], 0.86618862651486, 6)
        self.assertAlmostEqual(p[2], 0.00000022184455, 6)
        self.assertAlmostEqual(p[3], 0.00000017581142, 6)
        self.assertAlmostEqual(p[4], 0.00000021157430, 6)
        self.assertAlmostEqual(p[5], 0.00000027136657, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 24.33034492900383, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 24.42115827627032, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 44.99906123473563, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 44.99906123473501, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][4],dm1_exc[1][4]), 44.99906123473585, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][5],dm1_exc[1][5]), 44.99906123473656, 6)

    def test_ip_adc2x(self):

        myadc.ncvs = 2
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.073511478250083, 6)

        e,v,p,x = myadc.kernel(nroots=6)

        self.assertAlmostEqual(e[0], 7.28130873572764, 6)
        self.assertAlmostEqual(e[1], 7.30630109221758, 6)
        self.assertAlmostEqual(e[2], 7.44650004529215, 6)
        self.assertAlmostEqual(e[3], 7.51447307852239, 6)
        self.assertAlmostEqual(e[4], 7.51447307852240, 6)
        self.assertAlmostEqual(e[5], 7.51447307852271, 6)

        self.assertAlmostEqual(p[0], 0.84500751263818, 6)
        self.assertAlmostEqual(p[1], 0.83971631351728, 6)
        self.assertAlmostEqual(p[2], 0.01185532383567, 6)
        self.assertAlmostEqual(p[3], 0.00000140035322, 6)
        self.assertAlmostEqual(p[4], 0.00000140035298, 6)
        self.assertAlmostEqual(p[5], 0.00000140035318, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 24.071667937205092, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 24.195548010935678, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 27.598011511917026, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 27.735289852038065, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][4],dm1_exc[1][4]), 27.735289853609093, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][5],dm1_exc[1][5]), 27.735289844280143, 6)

    def test_ip_adc3_high_cost(self):

        myadc.ncvs = 2
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e,-0.08920409983024571 , 6)

        e,v,p,x = myadc.kernel(nroots=6)

        self.assertAlmostEqual(e[0], 7.31575950423717, 6)
        self.assertAlmostEqual(e[1], 7.34025236653769, 6)
        self.assertAlmostEqual(e[2], 7.44714737514894, 6)
        self.assertAlmostEqual(e[3], 7.51447307852241, 6)
        self.assertAlmostEqual(e[4], 7.51447307852241, 6)
        self.assertAlmostEqual(e[5], 7.51447307852259, 6)

        self.assertAlmostEqual(p[0], 0.86005126939397, 6)
        self.assertAlmostEqual(p[1], 0.84578527227102, 6)
        self.assertAlmostEqual(p[2], 0.02111404130930, 6)
        self.assertAlmostEqual(p[3], 0.00000140035297, 6)
        self.assertAlmostEqual(p[4], 0.00000140035305, 6)
        self.assertAlmostEqual(p[5], 0.00000140035318, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 24.30525761733679, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 24.46041808613336, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 27.57372942007785, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 27.73528982525924, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][4],dm1_exc[1][4]), 27.73528984901638, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][5],dm1_exc[1][5]), 27.73528988288061, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell atom")
    unittest.main()
