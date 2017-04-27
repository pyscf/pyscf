from __future__ import print_function
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.tools.siesta_utils import get_siesta_command, get_pseudo
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

print(siesta_fdf, file=open(label+'.fdf', 'w'))

for sp in ['Na']:  os.symlink(get_pseudo(sp), sp+'.psf')

errorcode = subprocess.call(get_siesta_command(label), shell=True)
if errorcode: raise RuntimeError('siesta returned an error: {0}'.format(errorcode))

sv  = system_vars_c(label) # run test system_vars
assert sv.norbs == 10
