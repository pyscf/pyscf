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
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc, myadc_fr
    mol = gto.Mole()
    r = 0.957492
    x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
    y = r * math.cos(104.468205* math.pi/(2 * 180.0))
    mol.atom = [
        ['O', (0., 0.    , 0)],
        ['H', (0., -x, y)],
        ['H', (0., x , y)],]
    mol.basis = {'H': 'cc-pVDZ',
                 'O': 'cc-pVDZ',}
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc_fr = adc.ADC(mf,frozen=1)
    myadc_fr.conv_tol = 1e-12
    myadc_fr.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 18.961583618234123, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4034634878946100, 6)
        self.assertAlmostEqual(e[1], 0.4908881395275673, 6)
        self.assertAlmostEqual(e[2], 0.6573303400764507, 6)

        self.assertAlmostEqual(p[0], 1.8159461951985487, 6)
        self.assertAlmostEqual(p[1], 1.8271278901200159, 6)
        self.assertAlmostEqual(p[2], 1.8580497651729584, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 14.403019200419338, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 14.332753419421664, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 14.137362375993561, 6)

    def test_ip_adc2x(self):
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 18.961583618234123, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4085610789192171, 6)
        self.assertAlmostEqual(e[1], 0.4949784593692911, 6)
        self.assertAlmostEqual(e[2], 0.6602619900185128, 6)

        self.assertAlmostEqual(p[0], 1.8294255723926962, 6)
        self.assertAlmostEqual(p[1], 1.8379855493632298, 6)
        self.assertAlmostEqual(p[2], 1.8668122877720497, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 14.540939668787368, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 14.461857735628373, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 14.254315012440417, 6)


    def test_ip_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2107769014592799, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 19.067653558507857, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=4)
        myadcip.analyze()

        self.assertAlmostEqual(e[0], 0.4481211042230935, 6)
        self.assertAlmostEqual(e[1], 0.5316292617891758, 6)
        self.assertAlmostEqual(e[2], 0.6850054080600295, 6)
        self.assertAlmostEqual(e[3], 1.1090318744878, 6)

        self.assertAlmostEqual(p[0], 1.8683046103300467, 6)
        self.assertAlmostEqual(p[1], 1.8720323166638342, 6)
        self.assertAlmostEqual(p[2], 1.888224692086022, 6)
        self.assertAlmostEqual(p[3], 0.1651156836207662, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 14.885572979960191, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 14.77082174697975, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 14.526107108653937, 6)

    def test_ip_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.2086469399105177, 6)

        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 19.06657244804087, 6)

        myadcip_fr = adc.radc_ip.RADCIP(myadc_fr)
        e,v,p,x = myadcip_fr.kernel(nroots=4)
        myadcip_fr.analyze()

        self.assertAlmostEqual(e[0], 0.44800668424168, 6)
        self.assertAlmostEqual(e[1], 0.53153771640387, 6)
        self.assertAlmostEqual(e[2], 0.68490751867029, 6)
        self.assertAlmostEqual(e[3], 1.10909800938983, 6)

        self.assertAlmostEqual(p[0], 1.8682840079407685, 6)
        self.assertAlmostEqual(p[1], 1.8720001297929703, 6)
        self.assertAlmostEqual(p[2], 1.8881913246364717, 6)
        self.assertAlmostEqual(p[3], 0.16536157361872156, 6)

        dm1_exc = myadcip_fr.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 14.884814733432336, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 14.769932232599087, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 14.525296424926589, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 21.678771476096745, 6)


    def test_ip_adc2_frozen(self):
        adc2 = adc.ADC(mf,frozen=1)
        adc2.method = 'adc(2)'
        e_exc, v_exc = adc2.kernel()[:2]
        self.assertAlmostEqual(e_exc[0], 0.40346356, 6)
        self.assertEqual(v_exc.shape, (308, 1))

        adc2 = adc.ADC(mf,frozen=np.array([0]))
        adc2.method = 'adc(2)'
        e_exc, v_exc = adc2.kernel()[:2]
        self.assertAlmostEqual(e_exc[0], 0.40346356, 6)
        self.assertEqual(v_exc.shape, (308, 1))

        adc2 = adc.ADC(mf,frozen=0)
        adc2.method = 'adc(2)'
        e_exc, v_exc = adc2.kernel()[:2]
        self.assertAlmostEqual(e_exc[0], 0.40346348, 6)
        self.assertEqual(v_exc.shape, (480, 1))

        adc2 = adc.ADC(mf,frozen=2)
        adc2.method = 'adc(2)'
        e_exc, v_exc = adc2.kernel()[:2]
        self.assertAlmostEqual(e_exc[0], 0.42796576, 6)
        self.assertEqual(v_exc.shape, (174, 1))

if __name__ == "__main__":
    print("IP calculations for different ADC methods for water molecule")
    unittest.main()
