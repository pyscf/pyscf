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

class KnowValues(unittest.TestCase):

  def test_water_pp_pb(self):
    """ This is for initializing with SIESTA radial orbitals """
    from pyscf.nao import system_vars_c, prod_basis_c
    from numpy import einsum, array
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', cd=dname)
    self.assertTrue(abs(sv.ucell).sum()>0)
    pb = prod_basis_c().init_prod_basis_pp(sv)
    self.assertEqual(sv.norbs, 23)
    
    self.assertEqual(len(pb.bp2info), 3)
    vden = pb.get_ac_vertex_array()
    ccden = pb.get_da2cc_den()
    moms = pb.comp_moments()

    oref = sv.overlap_coo().toarray()
    over = einsum('lab,l->ab', vden, moms[0])

    dcoo = sv.dipole_coo()
    dref = array([dc.toarray() for dc in dcoo])
    dipo = einsum('lab,lx->xab', vden, moms[1])
    
    emean = (abs(oref-over).sum()/(oref.size))
    emax  = (abs(oref-over).max())
    self.assertAlmostEqual(emean, 0.000102115844911, 4)
    self.assertAlmostEqual(emax, 0.00182562129245, 4)

    emean = (abs(dref-dipo).sum()/(dref.size))
    emax = (abs(dref-dipo).max())
    self.assertAlmostEqual(emean, 0.000618731257284, 4)
    self.assertAlmostEqual(emax, 0.0140744946617, 4)
    
#    print(vden.shape)
#    print(ccden.shape)
#    print(moms[0].shape)
#    print(moms[1].shape)
    

if __name__ == "__main__":
  unittest.main()
