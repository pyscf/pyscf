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

from __future__ import print_function
import unittest
f#rom pyscf.nao.m_siesta_utils import get_siesta_command, get_pseudo

class KnowValues(unittest.TestCase):
  
  def test_siesta2sv_df(self):
    import subprocess
    import os

    siesta_fdf = """
    SystemName          O2 
    SystemLabel         O2
    AtomicCoordinatesFormat  Ang
    NumberOfAtoms       2
    NumberOfSpecies     1    

    %block ChemicalSpeciesLabel
    1	8	O.gga
    %endblock ChemicalSpeciesLabel

    LatticeConstant     20 Ang

    %block LatticeVectors             
      1.000  0.000  0.000
      0.000  1.000  0.000
      0.000  0.000  1.000
    %endblock LatticeVectors

    %block AtomicCoordinatesAndAtomicSpecies
	  0.0  0.0  0.622978	1		1	O.gga
	  0.0  0.0  -0.622978	1		2	O.gga
    %endblock AtomicCoordinatesAndAtomicSpecies

    PAO.BasisSize       DZP             
    PAO.EnergyShift     10 meV         
    PAO.SplitNorm       0.15            
    MeshCutoff          250.0 Ry

    XC.functional        GGA           
    XC.authors           PBE           
    SpinPolarized       .true.

    SCFMustConverge     .true.
    DM.Tolerance         1.d-4         

    ElectronicTemperature      10 K
    MD.TypeOfRun               CG
    MD.NumCGsteps              0
    MaxSCFIterations           100

    WriteDenchar        .true.
    COOP.Write          .true.
    xml.write           .true.
    """


    fi = open('O2.fdf', 'w')
    print(siesta_fdf, file=fi)
    fi.close()
    
    subprocess.run("siesta < O2.fdf", shell=True)


    # run test system_vars
    #from pyscf.nao import mf
    #sv  = mf(label=label)
    #self.assertEqual(sv.norbs, 10)
    #self.assertTrue( sv.diag_check() )
    #self.assertTrue( sv.overlap_check())


    from pyscf.nao import gw as gw_c
    import os
    dname = os.getcwd()
    gw = gw_c(label='O2', cd=dname, verbosity=5, niter_max_ev=70, rescf=True,magnetization=2)
    gw.kernel_gw()
    gw.report()

if __name__ == "__main__": unittest.main()
