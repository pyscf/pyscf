#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import grad
from pyscf.grad import dhf

def setUpModule():
    global h2o
    h2o = gto.Mole()
    h2o.verbose = 5
    h2o.output = '/dev/null'
    h2o.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o.basis = {"H": '6-31g',
                 "O": '6-31g',}
    h2o.build()

def tearDownModule():
    global h2o
    h2o.stdout.close()
    del h2o


class KnownValues(unittest.TestCase):
    def test_dhf_grad_with_ssss_high_cost(self):
        with lib.light_speed(30):
            mf = scf.DHF(h2o).run(conv_tol=1e-12)
            g = mf.nuc_grad_method().kernel()
            self.assertAlmostEqual(lib.fp(g), 0.0074947016737157545, 6)

            ms = mf.as_scanner()
            pmol = h2o.copy()
            e1 = ms(pmol.set_geom_([["O" , (0. , 0.     ,-0.001)],
                                    [1   , (0. , -0.757 , 0.587)],
                                    [1   , (0. , 0.757  , 0.587)]], unit='Ang'))
            e2 = ms(pmol.set_geom_([["O" , (0. , 0.     , 0.001)],
                                    [1   , (0. , -0.757 , 0.587)],
                                    [1   , (0. , 0.757  , 0.587)]], unit='Ang'))
            self.assertAlmostEqual(g[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

#    def test_dhf_grad_without_ssss(self):
#        with lib.light_speed(30):
#            mf = scf.DHF(h2o).set(with_ssss=False).run()
#            g = mf.nuc_grad_method().kernel()  # NotImplemented
#            self.assertAlmostEqual(lib.fp(g), 0.035838032078025273, 7)
#
#            ms = mf.as_scanner()
#            pmol = h2o.copy()
#            e1 = ms(pmol.set_geom_([["O" , (0. , 0.     ,-0.001)],
#                                    [1   , (0. , -0.757 , 0.587)],
#                                    [1   , (0. , 0.757  , 0.587)]], unit='Ang'))
#            e2 = ms(pmol.set_geom_([["O" , (0. , 0.     , 0.001)],
#                                    [1   , (0. , -0.757 , 0.587)],
#                                    [1   , (0. , 0.757  , 0.587)]], unit='Ang'))
#            self.assertAlmostEqual(g[0,2], (e2-e1)/0.002*lib.param.BOHR, 6)


if __name__ == "__main__":
    print("Full Tests for DHF")
    unittest.main()
