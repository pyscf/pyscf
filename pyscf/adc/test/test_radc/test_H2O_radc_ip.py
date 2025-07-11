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
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
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

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

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
        self.assertAlmostEqual(r2_gs, 19.073700043115412, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4034634878946100, 6)
        self.assertAlmostEqual(e[1], 0.4908881395275673, 6)
        self.assertAlmostEqual(e[2], 0.6573303400764507, 6)

        self.assertAlmostEqual(p[0], 1.8162558898737797, 6)
        self.assertAlmostEqual(p[1], 1.8274312312239454, 6)
        self.assertAlmostEqual(p[2], 1.8582314560275948, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 14.495740213945693, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 14.420886528704559, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 14.222914438696922, 6)

    def test_ip_adc2x(self):
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 19.073700043115412, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4085610789192171, 6)
        self.assertAlmostEqual(e[1], 0.4949784593692911, 6)
        self.assertAlmostEqual(e[2], 0.6602619900185128, 6)

        self.assertAlmostEqual(p[0], 1.8296221555740104, 6)
        self.assertAlmostEqual(p[1], 1.8381884804163264, 6)
        self.assertAlmostEqual(p[2], 1.8669268953278064, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 14.63435450911693, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 14.55055095791920, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 14.34027603791598, 6)


    def test_ip_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2107769014592799, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 19.043496230938608, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=4)
        myadcip.analyze()

        self.assertAlmostEqual(e[0], 0.4481211042230935, 6)
        self.assertAlmostEqual(e[1], 0.5316292617891758, 6)
        self.assertAlmostEqual(e[2], 0.6850054080600295, 6)
        self.assertAlmostEqual(e[3], 1.1090318744878, 6)

        self.assertAlmostEqual(p[0], 1.8682367032338498, 6)
        self.assertAlmostEqual(p[1], 1.8720029748507658, 6)
        self.assertAlmostEqual(p[2], 1.8881842403480831, 6)
        self.assertAlmostEqual(p[3], 0.1651131053450, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 14.865794062106032, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 14.750656672998344, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 14.508101917384584, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for water molecule")
    unittest.main()
