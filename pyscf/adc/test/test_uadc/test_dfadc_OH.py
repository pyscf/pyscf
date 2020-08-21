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


r = 0.969286393
mol = gto.Mole()
mol.atom = [
    ['O', ( 0., 0.    , -r/2   )],
    ['H', ( 0., 0.    ,  r/2)],]
mol.basis = {'O':'cc-pvdz',
             'H':'cc-pvdz'}
mol.verbose = 0
mol.symmetry = False
mol.spin  = 1
mol.build()
mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-ri')
mf.conv_tol = 1e-12
mf.kernel()
myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_df_gs(self):
  
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.150971806035, 6)

    def test_ea_dfadc3(self):
  
        myadc.max_memory = 20
        myadc.method = "adc(3)"
        myadc.method_type = "ea"
        
        e,v,p = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.033539067176, 6)
        self.assertAlmostEqual(e[1], 0.172594837454, 6)
        self.assertAlmostEqual(e[2], 0.177948387177, 6)
        self.assertAlmostEqual(e[3], 0.202146181892, 6)

        self.assertAlmostEqual(p[0], 0.936537492273, 6)
        self.assertAlmostEqual(p[1], 0.984032111351, 6)
        self.assertAlmostEqual(p[2], 0.779532088815, 6)
        self.assertAlmostEqual(p[3], 0.204730732440, 6)


    def test_ip_dfadc3_dif_aux_basis(self):
  
        myadc.with_df = df.DF(mol, auxbasis='aug-cc-pvdz-ri')
        myadc.max_memory = 2
        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        
        e,v,p = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr        

        self.assertAlmostEqual(e_corr, -0.163313165072, 6)

        self.assertAlmostEqual(e[0], 0.457054316116, 6)
        self.assertAlmostEqual(e[1], 0.468154829875, 6)
        self.assertAlmostEqual(e[2], 0.556499969477, 6)

        self.assertAlmostEqual(p[0], 0.938692139911, 6)
        self.assertAlmostEqual(p[1], 0.586912661634, 6)
        self.assertAlmostEqual(p[2], 0.351125891139, 6)
      
if __name__ == "__main__":
    print("DF-ADC calculations for different RADC methods for OH")
    unittest.main()
