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
from pyscf import df

r = 1.098
mol = gto.Mole()
mol.atom = [
    ['N', ( 0., 0.    , -r/2   )],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'cc-pvdz'}
mol.verbose = 0
mol.build()
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf.conv_tol = 1e-12
mf.kernel()
myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):


    def test_df_gs(self):
  
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.310726976610, 6)

    def test_ip_dfadc3(self):
  
        myadc.max_memory = 2
        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        
        e,v,p = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr        

        self.assertAlmostEqual(e_corr,  -0.3060064030896477, 6)

        self.assertAlmostEqual(e[0], 0.556062477042, 6)
        self.assertAlmostEqual(e[1], 0.601086700687, 6)
        self.assertAlmostEqual(e[2], 0.601086700687, 6)

        self.assertAlmostEqual(p[0], 1.832574315125, 6)
        self.assertAlmostEqual(p[1], 1.863875684003, 6)
        self.assertAlmostEqual(p[2], 1.863875684003, 6)

    def test_ea_dfadc2_dif_aux_basis(self):
  
        myadc = adc.ADC(mf).density_fit(auxbasis='cc-pvdz-ri')
        myadc.max_memory = 20
        myadc.method = "adc(2)"
        myadc.method_type = "ea"
        
        e,v,p = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.142607731998, 6)
        self.assertAlmostEqual(e[1], 0.142607731998, 6)
        self.assertAlmostEqual(e[2], 0.550838451444, 6)
        self.assertAlmostEqual(e[3], 0.767365787595, 6)

        self.assertAlmostEqual(p[0], 1.866037962900, 6)
        self.assertAlmostEqual(p[1], 1.866037962900, 6)
        self.assertAlmostEqual(p[2], 1.926996353285, 6)
        self.assertAlmostEqual(p[3], 1.883660023372, 6)
      
if __name__ == "__main__":
    print("DF-ADC calculations for different RADC methods for nitrogen molecule")
    unittest.main()
