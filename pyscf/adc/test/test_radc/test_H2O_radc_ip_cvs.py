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
import numpy
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

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ip_cvs_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        myadcipcvs = adc.radc_ip_cvs.RADCIPCVS(myadc)
        myadcipcvs.ncvs = 1
        e,v,p,x = myadcipcvs.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 19.83739019952255, 6)

        self.assertAlmostEqual(p[0], 1.54937962073732, 6)

    def test_ip_adc2x(self):

        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        myadcipcvs = adc.radc_ip_cvs.RADCIPCVS(myadc)
        myadcipcvs.ncvs = 1
        e,v,p,x = myadcipcvs.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 19.86256087818720, 6)
        self.assertAlmostEqual(e[1], 21.24443090401234, 6)
        self.assertAlmostEqual(e[2], 21.27391652329249, 6)

        self.assertAlmostEqual(p[0], 1.57448682772367, 6)
        self.assertAlmostEqual(p[1], 0.00000138285407, 6)
        self.assertAlmostEqual(p[2], 0.00000284749463, 6)


    def test_ip_adc3(self):

        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2107769014592799, 6)

        myadcipcvs = adc.radc_ip_cvs.RADCIPCVS(myadc)
        myadcipcvs.ncvs = 1
        e,v,p,x = myadcipcvs.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 20.09352653091772, 6)
        self.assertAlmostEqual(e[1], 21.24443090413907, 6)
        self.assertAlmostEqual(e[2], 21.27391652292367, 6)

        self.assertAlmostEqual(p[0], 1.66994015437000, 6)
        self.assertAlmostEqual(p[1], 0.00000138285406, 6)
        self.assertAlmostEqual(p[2], 0.00000284749466, 6)

if __name__ == "__main__":
    print("IP-CVS calculations for different ADC methods for water molecule")
    unittest.main()
