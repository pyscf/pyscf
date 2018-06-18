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

  def test_fxc(self):
    """ Compute TDDFT interaction kernel  """
    from timeit import default_timer as timer

    from pyscf.nao import system_vars_c, prod_basis_c
    from pyscf.nao.m_comp_dm import comp_dm
    from timeit import default_timer as timer
    
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    pb = prod_basis_c().init_prod_basis_pp(sv)
    dm = comp_dm(sv.wfsx.x, sv.get_occupations())
    fxc = pb.comp_fxc_lil(dm, xc_code='1.0*LDA,1.0*PZ', level=4)

if __name__ == "__main__": unittest.main()
