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
import numpy
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

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):
        myadc.method = "adc(2)"
        myadc.approx_trans_moments = False
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2218560613160146, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        myadcea.approx_trans_moments = False
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.028767540957364002, 6)
        self.assertAlmostEqual(e[1], 0.055347550860014215, 6)
        self.assertAlmostEqual(e[2], 0.1643553772982312, 6)

        self.assertAlmostEqual(p[0],1.986819690876559, 6)
        self.assertAlmostEqual(p[1],1.9941128816000497 , 6)
        self.assertAlmostEqual(p[2],1.97604203301431 , 6)


    def test_ip_adc2(self):
        myadc.method = "adc(2)"
        myadc.approx_trans_moments = False
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2218560613160146, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        myadcip.approx_trans_moments = False
        e,v,p,x = myadcip.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4133257511136192, 6)
        self.assertAlmostEqual(e[1], 0.4978545288459139, 6)
        self.assertAlmostEqual(e[2], 0.660776973898569, 6)
        self.assertAlmostEqual(e[3], 1.0540027121314526, 6)

        self.assertAlmostEqual(p[0], 1.7709138169235759, 6)
        self.assertAlmostEqual(p[1], 1.7748323939182178, 6)
        self.assertAlmostEqual(p[2], 1.8039728206947119, 6)
        self.assertAlmostEqual(p[3], 0.003965957913241936, 6)


    def test_ea_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.approx_trans_moments = False
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.21936553835290695, 6)

        myadcea_fr = adc.radc_ea.RADCEA(myadc_fr)
        myadcea_fr.approx_trans_moments = False
        e,v,p,x = myadcea_fr.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0287640767713242, 6)
        self.assertAlmostEqual(e[1], 0.05534877596994694, 6)
        self.assertAlmostEqual(e[2], 0.16435779396607947, 6)

        self.assertAlmostEqual(p[0],1.986820069114505, 6)
        self.assertAlmostEqual(p[1],1.994114410418263 , 6)
        self.assertAlmostEqual(p[2],1.976045811220421 , 6)


    def test_ip_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.approx_trans_moments = False
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.21936553835290695, 6)

        myadcip_fr = adc.radc_ip.RADCIP(myadc_fr)
        myadcip_fr.approx_trans_moments = False
        e,v,p,x = myadcip_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.41331768842528704, 6)
        self.assertAlmostEqual(e[1], 0.4978944254083694, 6)
        self.assertAlmostEqual(e[2], 0.6607828254528624, 6)
        self.assertAlmostEqual(e[3], 1.0540042162192191, 6)

        self.assertAlmostEqual(p[0], 1.7709558649111785, 6)
        self.assertAlmostEqual(p[1], 1.774875940256063, 6)
        self.assertAlmostEqual(p[2], 1.8040072766943684, 6)
        self.assertAlmostEqual(p[3], 0.003934812008221318, 6)

if __name__ == "__main__":
    print("Exact transition moments calculations for different RADC methods for water molecule")
    unittest.main()
