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

  def test_mo_cube_libnao(self):
    """ Computing of the atomic orbitals """
    from pyscf.nao import system_vars_c
    from pyscf.tools.cubegen import Cube
   
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    cc = Cube(sv, nx=40, ny=40, nz=40)
    co2val = sv.comp_aos_den(cc.get_coords())
    nocc_0t = int(sv.nelectron / 2)
    c2orb =  np.dot(co2val, sv.wfsx.x[0,0,nocc_0t,:,0]).reshape((cc.nx, cc.ny, cc.nz))
    cc.write(c2orb, "water_mo.cube", comment='HOMO')
    
if __name__ == "__main__": unittest.main()
