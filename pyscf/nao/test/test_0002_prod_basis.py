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
from pyscf import gto
from pyscf.nao import system_vars_c, prod_log_c

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_gto2sv_df(self):
    from pyscf import scf
    """ Test import of density-fitting Gaussian functions ... hm """
    mf = scf.density_fit(scf.RHF(mol))
    self.assertAlmostEqual(mf.scf(), -76.025936299702536, 2)
    sv = system_vars_c().init_pyscf_gto(mol)
    prod_log = prod_log_c().init_prod_log_df(mf.with_df.auxmol, sv)
    self.assertEqual(prod_log.rr[0], sv.ao_log.rr[0])
    self.assertEqual(prod_log.pp[0], sv.ao_log.pp[0])
    self.assertEqual(prod_log.nspecies, sv.ao_log.nspecies)
    self.assertEqual(prod_log.sp2charge, sv.ao_log.sp2charge)
    #print(prod_log.sp_mu2rcut)
    #prod_log.view()
    #print(dir(mf))
    #print(dir(mf.mol))
    #print(dir(mf.with_df.auxmol))
    #print(mf.with_df.auxmol._basis)

  def test_gto2sv_prod_log(self):
    """ Test what ? """
    sv = system_vars_c().init_pyscf_gto(mol)
    prod_log = prod_log_c().init_prod_log_dp(sv.ao_log, tol_loc=1e-4)
    mae,mxe,lll=prod_log.overlap_check()
    self.assertTrue(all(lll))
    self.assertEqual(prod_log.nspecies, 2)
    self.assertEqual(prod_log.sp2nmult[0], 7)
    self.assertEqual(prod_log.sp2nmult[1], 20)
    self.assertEqual(prod_log.sp2norbs[0], 15)
    self.assertEqual(prod_log.sp2norbs[1], 70)
    
if __name__ == "__main__":
  print("Full Tests for prod_basis_c")
  unittest.main()

