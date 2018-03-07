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


class KnowValues(unittest.TestCase):
  
  def test_siesta2sv_df(self):
    from pyscf import scf
    from pyscf.nao.m_siesta_utils import get_siesta_command, get_pseudo
    import subprocess
    import os

    siesta_fdf = """
    xml.write                  .true.
    PAO.EnergyShift            100 meV
    %block ChemicalSpeciesLabel
     1  11  Na
    %endblock ChemicalSpeciesLabel
    
    NumberOfAtoms       2
    NumberOfSpecies     1
    %block AtomicCoordinatesAndAtomicSpecies
        0.77573521    0.00000000    0.00000000   1
       -0.77573521    0.00000000    0.00000000   1
    %endblock AtomicCoordinatesAndAtomicSpecies
    
    MD.NumCGsteps              0
    COOP.Write                 .true.
    WriteDenchar               .true.
    """
    label = 'siesta'
    fi = open(label+'.fdf', 'w')
    print(siesta_fdf, file=fi)
    fi.close()
    for sp in ['Na']: 
      try:
        os.remove(sp+'.psf')
      except :
        pass
      try:
        pppath = get_pseudo(sp)
      except:
        print('get_pseudo( '+sp+' ) is not working--> skip siesta run' )
        return
      os.symlink(pppath, sp+'.psf')
    errorcode = subprocess.call(get_siesta_command(label), shell=True)
    if errorcode: raise RuntimeError('siesta returned an error: {0}'.format(errorcode))

    # run test system_vars
    from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
    sv  = system_vars_c().init_siesta_xml(label)
    self.assertEqual(sv.norbs, 10)
    self.assertTrue( sv.diag_check() )
    self.assertTrue( sv.overlap_check())

if __name__ == "__main__":
  print("Full Tests for siesta-based sodium")
  unittest.main()
