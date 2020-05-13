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
from pyscf.nao import mf

class KnowValues(unittest.TestCase):

  def test_read_siesta_bulk_spin(self):
    """ Test reading of bulk, spin-resolved SIESTA calculation  """
    
    chdir = os.path.dirname(os.path.abspath(__file__))+'/ice'
    sv  = mf(label='siesta', cd=chdir, gen_pb=False)
    sv.diag_check()

if __name__ == "__main__": unittest.main()

