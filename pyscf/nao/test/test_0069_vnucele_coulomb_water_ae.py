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
from pyscf.nao import mf as mf_c
from pyscf import gto, scf, tddft
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0069_vnucele_coulomb_water_ae(self):
    """ This  """
    mol = gto.M(verbose=1,atom='O 0 0 0; H 0 0.489 1.074; H 0 0.489 -1.074',basis='cc-pvdz')
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    vne = mol.intor_symmetric('int1e_nuc')
    dm_gto = gto_mf.make_rdm1()
    E_ne_gto = (vne*dm_gto).sum()*0.5
    self.assertAlmostEqual(E_ne_gto, -97.67612964579993)

    tk = mol.intor_symmetric('int1e_kin')
    E_kin_gto = (tk*dm_gto).sum()
    self.assertAlmostEqual(E_kin_gto, 75.37551418889902)
    #print(__name__, E_ne_gto)
    
    mf = mf_c(gto=mol, mf=gto_mf)
    vne_nao = mf.vnucele_coo_coulomb().toarray()
    dm_nao = mf.make_rdm1().reshape((mf.norbs, mf.norbs))
    E_ne_nao = (vne_nao*dm_nao).sum()*0.5
    #print(__name__, E_ne_nao)
    self.assertAlmostEqual(E_ne_nao, -97.67612893873279)
    
    tk_nao = -0.5*mf.laplace_coo().toarray()
    E_kin_nao = (tk_nao*dm_nao).sum()
    #print(__name__, E_ne_nao)
    self.assertAlmostEqual(E_kin_nao, 75.37551418889902)
    
        
if __name__ == "__main__": unittest.main()
