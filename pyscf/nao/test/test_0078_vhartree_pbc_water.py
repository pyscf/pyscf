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
import os
import numpy as np
from timeit import default_timer as timer

from pyscf.data.nist import HARTREE2EV

from pyscf.nao import nao as nao_c
from pyscf.nao import mf as mf_c

class KnowValues(unittest.TestCase):

  def test_0078_build_3dgrid_pp_pbc_water(self):
    """ Test Hartree potential on equidistant grid with Periodic Boundary Conditions """
    dname = os.path.dirname(os.path.abspath(__file__))
    nao = nao_c(label='water', cd=dname, Ecut=100.0)
    gg,dv = nao.mesh3d.rr, nao.mesh3d.dv
    self.assertAlmostEqual(dv, 0.007621441417508375)
    self.assertEqual(len(gg), 3)
    self.assertTrue(gg[0].shape==(72,3))
    self.assertTrue(gg[1].shape==(54,3))
    self.assertTrue(gg[2].shape==(54,3))
    self.assertEqual(nao.mesh3d.size, 72*54*54)

  def test_0078_vhartree_pbc_water(self):
    """ Test Hartree potential on equidistant grid with Periodic Boundary Conditions """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='water', cd=dname, gen_pb=False, Ecut=100.0)
    d = abs(np.dot(mf.ucell_mom(), mf.ucell)-(2*np.pi)*np.eye(3)).sum()
    self.assertTrue(d<1e-15)
    g = mf.mesh3d.get_3dgrid()
    dens = mf.dens_elec(g.coords, mf.make_rdm1()).reshape(mf.mesh3d.shape)
    ts = timer()
    vh = mf.vhartree_pbc(dens)
    tf = timer()
    #print(__name__, tf-ts)
    E_Hartree = 0.5*(vh*dens*g.weights).sum()*HARTREE2EV
    self.assertAlmostEqual(E_Hartree, 382.8718239023864)
    # siesta:       Hartree =     382.890331

if __name__ == "__main__": unittest.main()
