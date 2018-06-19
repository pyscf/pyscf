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
from os.path import abspath, dirname
from pyscf.nao import mf
import unittest, numpy as np

sv = mf(label='water', cd=dirname(abspath(__file__)))
pb = sv.pb
vab_arr_ref = pb.get_dp_vertex_array()

class KnowValues(unittest.TestCase):

  def test_pb_doubly_sparse(self):
    """ This is to test generation of a doubly sparse format. """
    self.assertEqual(len(vab_arr_ref.shape), 3)
    self.assertEqual(vab_arr_ref.shape, (pb.npdp,pb.norbs,pb.norbs))
    vab_ds = pb.get_dp_vertex_doubly_sparse(axis=2)
    vab_arr = vab_ds.toarray()
    self.assertLess(abs(vab_arr-vab_arr_ref).sum()/vab_arr.size, 1e-15)

  def test_pb_sparse(self):
    """ This is to test generation of a doubly sparse format. """
    self.assertEqual(len(vab_arr_ref.shape), 3)
    self.assertEqual(vab_arr_ref.shape, (pb.npdp,pb.norbs,pb.norbs))
    vab_s = pb.get_dp_vertex_sparse()
    vab_arr = vab_s.toarray().reshape((pb.npdp,pb.norbs,pb.norbs))
    self.assertLess(abs(vab_arr-vab_arr_ref).sum()/vab_arr.size, 1e-15)
    
if __name__ == "__main__": unittest.main()

