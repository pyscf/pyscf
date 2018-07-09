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
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_aos_libnao(self):
    """ Computing of the atomic orbitals """
    from pyscf.nao import system_vars_c
    from pyscf.tools.cubegen import Cube
   
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    cc = Cube(sv, nx=20, ny=20, nz=20)
    aos = sv.comp_aos_den(cc.get_coords())
    self.assertEqual(aos.shape[0], cc.nx*cc.ny*cc.nz)
    self.assertEqual(aos.shape[1], sv.norbs)

if __name__ == "__main__": unittest.main()
