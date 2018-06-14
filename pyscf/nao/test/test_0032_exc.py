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

  def test_exc(self):
    """ Compute exchange-correlation energy """
    from timeit import default_timer as timer

    from pyscf.nao import mf
    from timeit import default_timer as timer
    
    sv = mf(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    dm = sv.make_rdm1()
    exc = sv.exc(dm, xc_code='1.0*LDA,1.0*PZ', level=4)
    #self.assertAlmostEqual(exc, -4.1422234271159333) ? redone water?
    self.assertAlmostEqual(exc, -4.1422239276270201)
    

if __name__ == "__main__": unittest.main()
