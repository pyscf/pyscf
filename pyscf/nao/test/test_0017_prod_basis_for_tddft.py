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
from pyscf.nao import mf
from numpy import allclose, float32, einsum

dname = os.path.dirname(os.path.abspath(__file__))
sv = mf(label='water', cd=dname)
pb = sv.pb

class KnowValues(unittest.TestCase):
  
  def test_vrtx_coo(self):
    """ This is to test the vertex in the sparse format """
    va = pb.get_dp_vertex_array()
    vc = pb.get_dp_vertex_sparse().toarray().reshape([pb.npdp,pb.norbs,pb.norbs])
    self.assertTrue(abs(va).sum()>0.0)
    self.assertTrue(allclose(vc, va))

    va = pb.get_dp_vertex_array(dtype=float32)
    vc = pb.get_dp_vertex_sparse(dtype=float32).toarray().reshape([pb.npdp,pb.norbs,pb.norbs])
    self.assertTrue(abs(va).sum()>0.0)
    self.assertTrue(allclose(vc, va))

  def test_vrtx_pab(self):
    """ This is to test the atom-centered vertex in the form of dense array """
    dab2v = pb.get_dp_vertex_array()
    dp2c = pb.get_da2cc_den()
    pab2v1 = einsum('dp,dab->pab', dp2c, dab2v)
    pab2v2 = pb.get_ac_vertex_array()
    self.assertTrue(allclose(pab2v1,pab2v2))

  def test_cc_coo(self):
    """ This is to test the gathering of conversion coefficients into a sparse format """
    cc_coo = pb.get_da2cc_sparse().toarray()
    cc_den = pb.get_da2cc_den()
    self.assertTrue(allclose(cc_coo, cc_den) )
    cc_coo = pb.get_da2cc_sparse(float32).toarray()
    cc_den = pb.get_da2cc_den(float32)
    self.assertTrue(allclose(cc_coo, cc_den) )    

if __name__ == "__main__": unittest.main()
