from __future__ import print_function
from pyscf.tools.siesta_utils import get_siesta_command, get_pseudo
import subprocess
import os

siesta_fdf = """
xml.write .true.
SpinPolarized .True. 
%block ChemicalSpeciesLabel
 1  8  O     
 2  1  H
%endblock ChemicalSpeciesLabel

AtomicCoordinatesFormat  Ang
NumberOfAtoms       3
NumberOfSpecies     2
%block AtomicCoordinatesAndAtomicSpecies
    0.00000000   -0.00164806    0.00000000   1       1  O
    0.77573521    0.59332141    0.00000000   2       2  H
   -0.77573521    0.59332141    0.00000000   2       3  H
%endblock AtomicCoordinatesAndAtomicSpecies

MD.NumCGsteps              0
MaxSCFIterations           100
COOP.Write     .true.
WriteDenchar   .true.
"""

label = 'siesta'

print(siesta_fdf, file=open(label+'.fdf', 'w'))

for sp in ['O', 'H']:  os.symlink(get_pseudo(sp), sp+'.psf')

errorcode = subprocess.call(get_siesta_command(label), shell=True)
if errorcode: raise RuntimeError('siesta returned an error: {0}'.format(errorcode))

# run test system_vars
from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
sv  = system_vars_c().init_siesta_xml(label = label)
assert sv.norbs == 23
assert diag_check(sv)
assert overlap_check(sv)
