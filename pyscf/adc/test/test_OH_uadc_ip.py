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
from pyscf import gto
from pyscf import scf
from pyscf import adc

r = 0.969286393
mol = gto.Mole()
mol.atom = [
    ['O', ( 0., 0.    , -r/2   )],
    ['H', ( 0., 0.    ,  r/2)],]
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

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):
  
        e, t_amp1, t_amp2 = myadc.kernel()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p = myadc.ip_adc(nroots=3)

        self.assertAlmostEqual(e[0], 0.548647707577121, 6)
        self.assertAlmostEqual(e[1], 0.5805631452816042, 6)
        self.assertAlmostEqual(e[2], 0.6211605152474929, 6)

        self.assertAlmostEqual(p[0], 0.8959919898235896, 6)
        self.assertAlmostEqual(p[1], 0.9119655743441659, 6)
        self.assertAlmostEqual(p[2], 0.90734152712232, 6)

    def test_ip_adc2x(self):
  
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p = myadc.ip_adc(nroots=3)

        self.assertAlmostEqual(e[0], 0.5336491676572919, 6)
        self.assertAlmostEqual(e[1], 0.5854392195128417, 6)
        self.assertAlmostEqual(e[2], 0.5994171901922393, 6)

        self.assertAlmostEqual(p[0], 0.7137923499030876, 6)
        self.assertAlmostEqual(p[1], 0.9230274901915541, 6)
        self.assertAlmostEqual(p[2], 0.6566547543204048, 6)

    def test_ip_adc3(self):
  
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel()
        self.assertAlmostEqual(e, -0.17616203329072194, 6)

        e,v,p = myadc.ip_adc(nroots=3)

        self.assertAlmostEqual(e[0], 0.5655300517024375, 6)
        self.assertAlmostEqual(e[1], 0.6199415726500781, 6)
        self.assertAlmostEqual(e[2], 0.623405730376162, 6)

        self.assertAlmostEqual(p[0], 0.3997295222107412, 6)
        self.assertAlmostEqual(p[1], 0.5083250051440014, 6)
        self.assertAlmostEqual(p[2], 0.931932697099881, 6)
      
if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
