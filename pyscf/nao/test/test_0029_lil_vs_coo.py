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
import os,unittest
from pyscf.nao import system_vars_c

sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))

class KnowValues(unittest.TestCase):

  def test_lil_vs_coo(self):
    """ Init system variables on libnao's site """
    lil = sv.overlap_lil().tocsr()
    coo = sv.overlap_coo().tocsr()
    derr = abs(coo-lil).sum()/coo.nnz
    self.assertLess(derr, 1e-12)

if __name__ == "__main__": unittest.main()
