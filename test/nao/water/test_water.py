from pyscf.nao.m_system_vars import system_vars_c
from pyscf.tools.siesta_utils import get_siesta_command, get_pseudo
import sys
import os
import subprocess

siesta_fdf = """
SystemName          Water 

PAO.EnergyShift 100 meV

SpinPolarized .True. 

%block ChemicalSpeciesLabel
 1  8  O     
 2  1  H
%endblock ChemicalSpeciesLabel

AtomicCoordinatesFormat  Ang
AtomCoorFormatOut Bohr

NumberOfAtoms       3
NumberOfSpecies     2

%block AtomicCoordinatesAndAtomicSpecies
    0.00000000   -0.00164806    0.00000000   1       1  O
    0.77573521    0.59332141    0.00000000   2       2  H
   -0.77573521    0.59332141    0.00000000   2       3  H
%endblock AtomicCoordinatesAndAtomicSpecies

### Molecular dynamics 
ElectronicTemperature      100 K
MD.TypeOfRun               CG
MD.NumCGsteps              1000
MaxSCFIterations           100

# Write .WFSX and .HSX files
COOP.Write     .true.
# Write .DIM and .PLD files
WriteDenchar   .true.
"""


label = 'siesta'

# write siesta input
f = open(label+'.fdf', 'w')
f.write(siesta_fdf)
f.close()

command = get_siesta_command(label)

# link to the psudo potential
for sp in ['O', 'H']:
    pseudo = get_pseudo(sp)
    os.symlink(pseudo, sp+'.psf')

# run siesta
errorcode = subprocess.call(command, shell=True)
if errorcode:
    raise RuntimeError('siesta returned an error: {0}'.format(errorcode))

# run test system_vars
sv  = system_vars_c(label)
assert sv.norbs == 23
