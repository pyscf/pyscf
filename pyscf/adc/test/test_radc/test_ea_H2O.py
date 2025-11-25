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
    mol.basis = {'H': 'aug-cc-pVDZ',
                 'O': 'aug-cc-pVDZ',}
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

    def test_ea_adc2(self):
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2218560609876961, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 20.3557963972156, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0287675413010661, 6)
        self.assertAlmostEqual(e[1], 0.0553475511361251, 6)
        self.assertAlmostEqual(e[2], 0.1643553780332306, 6)

        self.assertAlmostEqual(p[0], 1.9868196915945326, 6)
        self.assertAlmostEqual(p[1], 1.9941128865405613, 6)
        self.assertAlmostEqual(p[2], 1.9760420333383126, 6)

        dm1_exc = np.array(myadcea.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 60.74982115808177, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 77.50057807557833, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 51.451460124678604, 6)

    def test_ea_adc2x(self):
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2218560609876961, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 20.3557963972156, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0270276135717527, 6)
        self.assertAlmostEqual(e[1], 0.0546446308721235, 6)
        self.assertAlmostEqual(e[2], 0.1614552196278816, 6)

        self.assertAlmostEqual(p[0], 1.9782643804856972, 6)
        self.assertAlmostEqual(p[1], 1.9905409664546319, 6)
        self.assertAlmostEqual(p[2], 1.9593142553574816, 6)

        dm1_exc = myadcea.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 60.32114764303384, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 77.23400300967357, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 52.055954747398014, 6)


    def test_ea_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2263968409281272, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 20.070653779918267, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.0277406670820452, 6)
        self.assertAlmostEqual(e[1], 0.0551456657778995, 6)
        self.assertAlmostEqual(e[2], 0.1620710279026066, 6)
        self.assertAlmostEqual(e[3], 0.1882010099486046, 6)

        self.assertAlmostEqual(p[0], 1.9814233118436899, 6)
        self.assertAlmostEqual(p[1], 1.9920778842193207, 6)
        self.assertAlmostEqual(p[2], 1.9676462978544356, 6)
        self.assertAlmostEqual(p[3], 1.9743650630026532, 6)

        dm1_exc = myadcea.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 60.22057800411394, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 77.03749070234556, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 51.28103237617329, 6)


    def test_ea_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.21936553835290462, 6)

        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 20.3548798093427, 6)

        myadcea_fr = adc.radc_ea.RADCEA(myadc_fr)
        e,v,p,x = myadcea_fr.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0287640767713019, 6)
        self.assertAlmostEqual(e[1], 0.0553487759699679, 6)
        self.assertAlmostEqual(e[2], 0.1643577939660435, 6)

        self.assertAlmostEqual(p[0], 1.9868200691145088, 6)
        self.assertAlmostEqual(p[1], 1.9941144104182609, 6)
        self.assertAlmostEqual(p[2], 1.9760458112204118, 6)

        dm1_exc = np.array(myadcea_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 60.7471046759578, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 77.5003451983682, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 51.4529197597908, 6)


    def test_ea_adc2x_frozen(self):
        myadc_fr.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.21936553835290462, 6)

        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 20.3548798093427, 6)

        myadcea_fr = adc.radc_ea.RADCEA(myadc_fr)
        e,v,p,x = myadcea_fr.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0270230223088112, 6)
        self.assertAlmostEqual(e[1], 0.0546451436459182, 6)
        self.assertAlmostEqual(e[2], 0.1614545281030789, 6)

        self.assertAlmostEqual(p[0], 1.9782626731194528, 6)
        self.assertAlmostEqual(p[1], 1.9905413551544142, 6)
        self.assertAlmostEqual(p[2], 1.9593139012649046, 6)

        dm1_exc = np.array(myadcea_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 60.3179872699728, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 77.2333109653418, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 52.0575206212114, 6)

    def test_ea_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.2241204658960519, 6)

        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 20.07173839684320, 6)

        myadcea_fr = adc.radc_ea.RADCEA(myadc_fr)
        e,v,p,x = myadcea_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.0277386102474679, 6)
        self.assertAlmostEqual(e[1], 0.0551483888081231, 6)
        self.assertAlmostEqual(e[2], 0.1620802624386506, 6)
        self.assertAlmostEqual(e[3], 0.1882099906975863, 6)

        self.assertAlmostEqual(p[0], 1.9814229584714018, 6)
        self.assertAlmostEqual(p[1], 1.9920784490143197, 6)
        self.assertAlmostEqual(p[2], 1.9676456097852035, 6)
        self.assertAlmostEqual(p[3], 1.9743683769537101, 6)

        dm1_exc = np.array(myadcea_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 60.2204108853800, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 77.0399670701813, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 51.2838448114322, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 43.3784016488027, 6)

if __name__ == "__main__":
    print("EA calculations for different RADC methods for water molecule")
    unittest.main()
