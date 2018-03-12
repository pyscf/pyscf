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
import numpy as np
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

  def test_ao_eval(self):
    from pyscf.nao import system_vars_c
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao
    from pyscf.nao.m_ao_eval import ao_eval
    """  """
    sv = system_vars_c().init_pyscf_gto(mol)
    ra = np.array([2.0333, 0.101, 2.333])
    coords = np.array([[0.0333, 1.111, 3.333]])
    ao_vals_lib = ao_eval_libnao(sv.ao_log, ra, 0, coords)
    self.assertAlmostEqual(ao_vals_lib[0,0], 0.021725938009701302)
    ao_vals_lib = ao_eval_libnao(sv.ao_log, ra, 1, coords)
    self.assertAlmostEqual(ao_vals_lib[1,0], 0.0017709123325328384)
    
    ra = 4.0*np.random.rand(3)
    coords = 3.0*np.random.rand(10,3)
    ao_vals_lib = ao_eval_libnao(sv.ao_log, ra, 0, coords)
    ao_vals_py  = ao_eval(sv.ao_log, ra, 0, coords)
    for aocc1, aocc2 in zip(ao_vals_lib, ao_vals_py):
      for ao1, ao2 in zip(aocc1, aocc2):
        self.assertAlmostEqual(ao1, ao2)

if __name__ == "__main__":
  unittest.main()
