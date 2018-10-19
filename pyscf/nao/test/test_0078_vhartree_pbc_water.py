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
from pyscf.nao import nao as nao_c
from pyscf.nao import mf as mf_c
import os
import numpy as np

class KnowValues(unittest.TestCase):

  def test_0078_build_3dgrid_pp_pbc_water(self):
    """ Test Hartree potential on equidistant grid with Periodic Boundary Conditions """
    dname = os.path.dirname(os.path.abspath(__file__))
    nao = nao_c(label='water', cd=dname)
    gg,dv = nao.build_3dgrid_pp_pbc(Ecut=100.0)
    self.assertAlmostEqual(dv, 0.00802318938868)
    self.assertEqual(len(gg), 3)
    self.assertTrue(gg[0].shape==(72,1,1,3))
    self.assertTrue(gg[1].shape==(1,54,1,3))
    self.assertTrue(gg[2].shape==(1,1,54,3))

  def test_0078_vhartree_pbc_water(self):
    """ Test Hartree potential on equidistant grid with Periodic Boundary Conditions """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='water', cd=dname, gen_pb=False)
    d = abs(np.dot(mf.ucell_mom(), mf.ucell)-(2*np.pi)*np.eye(3)).sum()
    self.assertTrue(d<1e-15)

    vh,dv = mf.vhartree_pbc()
    print(__name__, dv, vh.sum()*dv)
    
if __name__ == "__main__": unittest.main()
