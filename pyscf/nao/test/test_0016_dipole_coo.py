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

from __future__ import print_function, division
import unittest
from pyscf import gto

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_dipole_coo(self):
    """ Test computation of dipole matrix elements """
    from pyscf.nao import system_vars_c
    sv = system_vars_c().init_pyscf_gto(mol)
    dipme = sv.dipole_coo()
    
    self.assertAlmostEqual(dipme[0].sum(), 23.8167121803)
    self.assertAlmostEqual(dipme[1].sum(), 18.9577251654)
    self.assertAlmostEqual(dipme[2].sum(), 48.1243277097)

#    self.assertAlmostEqual(dipme[0].sum(), 23.816263714841725)
#    self.assertAlmostEqual(dipme[1].sum(), 18.958562546276568)    
#    self.assertAlmostEqual(dipme[2].sum(), 48.124023241543377)

if __name__ == "__main__":
  unittest.main()
