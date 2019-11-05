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
from pyscf.nao import nao as nao_c
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0076_vna_vnl_Li(self):
    """ Test of the energy decomposition """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    nao = nao_c(label='li', cd=dname)
    n = nao.norbs
    dm  = nao.make_rdm1().reshape((n,n))
    vna = nao.vna_coo().toarray()

    Ena = (vna*dm).sum()*(-0.5)*HARTREE2EV
    self.assertAlmostEqual(Ena, 0.97907053041185432)
    #siesta: Ena     =         4.136159
    
    vnl = nao.vnl_coo().toarray()
    Enl = (vnl*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Enl, 1.3120833331377273)
    #siesta: Enl     =         1.312064

    vkin = -0.5*nao.laplace_coo().toarray() # Why not -0.5*Laplace ?
    Ekin = (vkin*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Ekin, 2.6283947063428021)
    #siesta: Ekin     =       2.628343

#siesta: Ebs     =        -2.302303
#siesta: Eions   =         9.635204
#siesta: Ena     =         4.136159
#siesta: Ekin    =         2.628343
#siesta: Enl     =         1.312064
#siesta: DEna    =         0.006680
#siesta: DUscf   =         0.000010
#siesta: DUext   =         0.000000
#siesta: Exc     =        -4.225864

if __name__ == "__main__": unittest.main()
