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
from pyscf.nao import nao as nao_c, scf as scf_c
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0075_vna_vnl_H2O(self):
    """ Test of the energy decomposition """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    nao = nao_c(label='water', cd=dname)
    n = nao.norbs
    dm  = nao.make_rdm1().reshape((n,n))
    vna = nao.vna_coo().toarray()

    Ena = (vna*dm).sum()*(-0.5)*HARTREE2EV
    self.assertAlmostEqual(Ena, 132.50585488810401)
    #siesta: Ena     =       175.007584
    
    vnl = nao.vnl_coo().toarray()
    Enl = (vnl*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Enl, -62.176213752828893)
    #siesta: Enl     =       -62.176200

    vkin = -0.5*nao.laplace_coo().toarray() # Why not -0.5*Laplace ?
    Ekin = (vkin*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Ekin, 351.76677461783862)
    #siesta: Ekin     =       351.769106 

#siesta: Ebs     =      -103.137894
#siesta: Eions   =       815.854478
#siesta: Ena     =       175.007584
#siesta: Ekin    =       351.769106
#siesta: Enl     =       -62.176200
#siesta: DEna    =        -2.594518
#siesta: DUscf   =         0.749718
#siesta: DUext   =         0.000000
    

if __name__ == "__main__": unittest.main()
