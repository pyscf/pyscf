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

class KnowValues(unittest.TestCase):

  def test_0077_chlocal_H2O(self):
    """ Test chlocal field (density of bare atoms) """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    nao = nao_c(label='water', cd=dname)
    g = nao.build_3dgrid_ae(level=3)
    int_chlocal = (g.weights*nao.vna(g.coords, sp2v=nao.ao_log.sp2chlocal)).sum()
    self.assertAlmostEqual(int_chlocal, -7.9999819496898787)

if __name__ == "__main__": unittest.main()
