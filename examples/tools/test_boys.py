#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: March 4, 2015
#
# Test file to illustrate the usage of boys localization
#

from pyscf import gto, scf
#from pyscf.tools import molden
#from pyscf.tools import localizer
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
mol.verbose = 4
mol.build()
mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

from pyscf.lo import boys
import numpy
mo = mf.mo_coeff[:,[17, 20, 21, 22, 23, 30, 36, 41, 42, 47, 48, 49]]
nmo = mo.shape[1]
u = numpy.linalg.svd(numpy.eye(nmo)+1e-5*numpy.random.random((nmo,nmo)))[0]
boys.Boys(mol).kernel(mo.dot(u), verbose=4)

#filename_mo   = 'benzene-631g-mo.molden'
#filename_boys = 'benzene-631g-boys.molden'
#
#with open( filename_mo, 'w' ) as thefile:
#    molden.header( mol, thefile )
#    molden.orbital_coeff( mol, thefile, mf.mo_coeff )
#print("Molecular orbitals saved in", filename_mo)
#
## Localize the pi-type orbitals. Counting starts from 0! 12 orbitals as 6-31G is DZ.
#tolocalize = np.array([17, 20, 21, 22, 23, 30, 36, 41, 42, 47, 48, 49]) - 1
#loc  = localizer.localizer( mol, mf.mo_coeff[:,tolocalize], 'boys' )
#loc.verbose = param.VERBOSE_DEBUG
#new_coeff = loc.optimize()
#loc.dump_molden( filename_boys, new_coeff )
#print("Boys localized pi-orbitals saved in", filename_boys)
