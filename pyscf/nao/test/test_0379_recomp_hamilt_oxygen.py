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

  def test_0379_recomp_hamilt_oxygen(self):
    """ Recomputing all parts of KS Hamiltonian a la SIESTA """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    F = -0.27012188215107336
    mf = mf_c(verbosity=1, label='oxygen', cd=dname, gen_pb=False, fermi_energy=F, Ecut=125.0)
    
    mf.overlap_check()
    mf.diag_check()

    dm = mf.make_rdm1().reshape((mf.norbs, mf.norbs))
    hamilt1 = mf.get_hamiltonian()[0].toarray()
    Ebs1 = (hamilt1*dm).sum()*HARTREE2EV
    #self.assertAlmostEqual(Ebs1, -73.18443295100552)

    g = mf.mesh3d.get_3dgrid()
    dens = mf.dens_elec(g.coords, dm=mf.make_rdm1()).reshape(mf.mesh3d.shape)
    mf.mesh3d.write('0379_dens_elec.cube', mol=mf, field=dens, comment='density')
    vh3d = mf.vhartree_pbc(dens)
    Ehartree = (vh3d*dens).sum()*0.5*HARTREE2EV*g.weights
    #self.assertAlmostEqual(Ehartree, 225.386052971981)

    vh = mf.vhartree_pbc_coo().toarray()
    Ehartree = (vh*dm).sum()*0.5*HARTREE2EV
    #self.assertAlmostEqual(Ehartree, 225.386052971981)

    tkin = -0.5*mf.laplace_coo().toarray()
    Ekin = (tkin*dm).sum()*HARTREE2EV
    #self.assertAlmostEqual(Ekin, 308.28672736957884)

    dens_atom = mf.vna(g.coords, sp2v=mf.ao_log.sp2chlocal, sp2rcut=mf.ao_log.sp2rcut_chlocal).reshape(mf.mesh3d.shape)
    dvh3d = mf.vhartree_pbc(dens+dens_atom)
    mf.mesh3d.write('0379_dvh.cube', mol=mf, field=dvh3d, comment='dVH')

    dvh  = mf.vhartree_pbc_coo(density_factors=[1,1]).toarray()
    Edvh = (dvh*dm).sum()*0.5*HARTREE2EV
    #self.assertAlmostEqual(Edvh, -109.5022234053683) # ???

    vhatom  = mf.vhartree_pbc_coo(density_factors=[0,1]).toarray()
    Ena_atom = (vhatom*dm).sum()*0.5*HARTREE2EV
    #self.assertAlmostEqual(Ena_atom, -334.88827637734926) # ???
    
    Exc  =  mf.exc()*HARTREE2EV
    #self.assertAlmostEqual(Exc, -87.78500822036436)  # Exc     =       -87.785701
    vxc_lil = mf.vxc_lil()
    vxc  = vxc_lil[0].toarray()

    vna  = mf.vna_coo().toarray()
    Ena = (vna*dm).sum()*HARTREE2EV
    #self.assertAlmostEqual(Ena, -217.77241233140285) # Enascf  =      -217.774110

    vnl  = mf.vnl_coo().toarray()
    Enl = (vnl*dm).sum()*HARTREE2EV
    #self.assertAlmostEqual(Enl, -43.272671826080604) # Enl     =       -43.273074

    for f1 in [1.0]:
      for f2 in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        for f3 in [1.0]:
          for f4 in [1.0]:
            for f5 in [1.0]:
              hamilt2 = f1*tkin + f2*vna + f3*vnl + f4*dvh + f5*vxc
              Ebs2 = (hamilt2*dm).sum()*HARTREE2EV
              print(__name__, f2, Ebs2)
              if abs(Ebs2+73.185007)<10.0:
                print(f1,f2,f3,f4,f5, Ebs2, -73.185007, Ebs2+73.185007, abs(hamilt2-hamilt1).sum())
    
    #self.assertAlmostEqual(Ebs2, -103.137894)

#siesta: Program's energy decomposition (eV):
#siesta: Ebs     =       -73.185007
#siesta: Eions   =       776.439835
#siesta: Ena     =       167.888678
#siesta: Enaatm  =      -222.571023
#siesta: Enascf  =      -217.774110
#siesta: Ekin    =       308.294510
#siesta: Enl     =       -43.273074
#siesta: DEna    =         4.796913
#siesta: DUscf   =         0.049454
#siesta: DUext   =         0.000000
#siesta: Exc     =       -87.785701
#siesta: eta*DQ  =         0.000000
#siesta: Emadel  =         0.000000
#siesta: Emeta   =         0.000000
#siesta: Emolmec =         0.000000
#siesta: Ekinion =         0.000000
#siesta: Eharris =      -426.469056
#siesta: Etot    =      -426.469056
#siesta: FreeEng =      -426.567787

#siesta: Final energy (eV):
#siesta:  Band Struct. =     -73.185007
#siesta:       Kinetic =     308.294510
#siesta:       Hartree =     225.387853
#siesta:    Ext. field =       0.000000
#siesta:   Exch.-corr. =     -87.785701
#siesta:  Ion-electron =    -717.206200
#siesta:       Ion-ion =    -155.159517
#siesta:       Ekinion =       0.000000
#siesta:         Total =    -426.469056

    
if __name__ == "__main__": unittest.main()
