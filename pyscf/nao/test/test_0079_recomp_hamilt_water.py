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

from pyscf.nao import mf as mf_c

class KnowValues(unittest.TestCase):

  def test_0079_recomp_hamilt_water(self):
    """ Recomputing all parts of KS Hamiltonian a la SIESTA """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='water', cd=dname, gen_pb=False, Ecut=100.0)
    dm = mf.make_rdm1().reshape((mf.norbs, mf.norbs))

    hamilt1 = mf.get_hamiltonian()[0].toarray()
    Ebs1 = (hamilt1*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Ebs1, -103.13708410565928)

    vh = mf.vhartree_pbc_coo().toarray()
    Ehartree = (vh*dm).sum()*0.5*HARTREE2EV
    self.assertAlmostEqual(Ehartree, 382.8718239023866)

    tkin = -0.5*mf.laplace_coo().toarray()
    Ekin = (tkin*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Ekin, 351.7667746178386)

    dvh  = mf.vhartree_pbc_coo(add_neutral_atom_density=True).toarray()
    Edvh = (dvh*dm).sum()*0.5*HARTREE2EV
    self.assertAlmostEqual(Edvh, -120.65336476645524)
    
    Exc  =  mf.exc()*HARTREE2EV
    self.assertAlmostEqual(Exc, -112.71570171625233)
    vxc_lil = mf.vxc_lil()
    vxc  = vxc_lil[0].toarray()

    vna  = mf.vna_coo().toarray()
    #Ena = (vna*dm).sum()*HARTREE2EV
    #self.assertAlmostEqual(Ena, 175.007584)

    vnl  = mf.vnl_coo().toarray()
    Enl = (vnl*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Enl, -62.17621375282889)

    hamilt2 = tkin + vna + vnl + dvh + vxc
    print(__name__)
    print(abs(hamilt1-hamilt2).sum() )
    print(abs(hamilt1+hamilt2).sum() )
    print(abs(hamilt1).sum() )
    print(abs(hamilt2).sum() )
    Ebs2 = (hamilt2*dm).sum()*HARTREE2EV
    self.assertAlmostEqual(Ebs2, -103.137894)

     
    

#siesta: Ebs     =      -103.137894
#siesta: Eions   =       815.854478
#siesta: Ena     =       175.007584
#siesta: Ekin    =       351.769106
#siesta: Enl     =       -62.176200
#siesta: DEna    =        -2.594518
#siesta: DUscf   =         0.749718
#siesta: DUext   =         0.000000
#siesta: Exc     =      -112.738374
#siesta: eta*DQ  =         0.000000
#siesta: Emadel  =         0.000000
#siesta: Emeta   =         0.000000
#siesta: Emolmec =         0.000000
#siesta: Ekinion =         0.000000
#siesta: Eharris =      -465.837162
#siesta: Etot    =      -465.837162
#siesta: FreeEng =      -465.837162

#siesta: Final energy (eV):
#siesta:  Band Struct. =    -103.137894
#siesta:       Kinetic =     351.769106
#siesta:       Hartree =     382.890331
#siesta:    Ext. field =       0.000000
#siesta:   Exch.-corr. =    -112.738374
#siesta:  Ion-electron =   -1072.976275
#siesta:       Ion-ion =     -14.781951
#siesta:       Ekinion =       0.000000
#siesta:         Total =    -465.837162
    
if __name__ == "__main__": unittest.main()
