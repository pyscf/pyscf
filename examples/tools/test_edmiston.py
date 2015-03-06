#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: March 5, 2015
#
# Test file to illustrate the usage of Edmiston-Ruedenberg localization
#

from pyscf import gto, scf
from pyscf.tools import molden
from pyscf.tools import localizer
from pyscf.lib import parameters as param
import numpy as np

mol = gto.Mole() # Benzene
mol.atom = '''
     H    0.000000000000     2.491406946734     0.000000000000
     C    0.000000000000     1.398696930758     0.000000000000
     H    0.000000000000    -2.491406946734     0.000000000000
     C    0.000000000000    -1.398696930758     0.000000000000
     H    2.157597486829     1.245660462400     0.000000000000
     C    1.211265339156     0.699329968382     0.000000000000
     H    2.157597486829    -1.245660462400     0.000000000000
     C    1.211265339156    -0.699329968382     0.000000000000
     H   -2.157597486829     1.245660462400     0.000000000000
     C   -1.211265339156     0.699329968382     0.000000000000
     H   -2.157597486829    -1.245660462400     0.000000000000
     C   -1.211265339156    -0.699329968382     0.000000000000
  '''
mol.basis = '6-31g'
mol.symmetry = 0
mol.charge = 0
mol.spin = 0
mol.build()
mf = scf.RHF( mol )
mf.verbose = 0
mf.scf()

filename_mo       = 'benzene-631g-mo.molden'
filename_edmiston = 'benzene-631g-edmiston.molden'

with open( filename_mo, 'w' ) as thefile:
    molden.header( mol, thefile )
    molden.orbital_coeff( mol, thefile, mf.mo_coeff )
print("Molecular orbitals saved in", filename_mo)

# Localize the pi-type orbitals. Counting starts from 0! 12 orbitals as 6-31G is DZ.
tolocalize = np.array([17, 20, 21, 22, 23, 30, 36, 41, 42, 47, 48, 49]) - 1
loc  = localizer.localizer( mol, mf.mo_coeff[:,tolocalize], 'edmiston' )
loc.verbose = param.VERBOSE_DEBUG
new_coeff = loc.optimize()
loc.dump_molden( filename_edmiston, new_coeff )
print("Edmiston-Ruedenberg localized pi-orbitals saved in", filename_edmiston)

