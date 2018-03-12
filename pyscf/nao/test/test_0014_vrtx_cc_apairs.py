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
import unittest,os


class KnowValues(unittest.TestCase):

  def test_vrtx_cc_apairs(self):
    """ This is to test a batch generation vertices for bilocal atomic pairs. """
    from pyscf.nao import system_vars_c, prod_basis_c
    from numpy import allclose

    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', cd=dname)
    pbb = prod_basis_c().init_prod_basis_pp_batch(sv)
    pba = prod_basis_c().init_prod_basis_pp(sv)

    for a,b in zip(pba.bp2info,pbb.bp2info):
      for a1,a2 in zip(a.atoms,b.atoms): self.assertEqual(a1,a2)
      for a1,a2 in zip(a.cc2a, b.cc2a): self.assertEqual(a1,a2)
      self.assertTrue(allclose(a.vrtx, b.vrtx))
      self.assertTrue(allclose(a.cc, b.cc))

    self.assertLess(abs(pbb.get_da2cc_sparse().tocsr()-pba.get_da2cc_sparse().tocsr()).sum(), 1e-9)
    self.assertLess(abs(pbb.get_dp_vertex_sparse().tocsr()-pba.get_dp_vertex_sparse().tocsr()).sum(), 1e-10)
    
if __name__ == "__main__": unittest.main()
