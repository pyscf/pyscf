#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: May 18, 2015
#
# Test file to illustrate the usage of the rhf_NewtonRaphson tool
#
# The gradient and hessian were determined from the equations in
# http://sebwouters.github.io/CheMPS2/doxygen/classCheMPS2_1_1CASSCF.html
# by throwing out all active space components.
#

import sys
from pyscf import gto, scf
from pyscf.tools import rhf_newtonraphson
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
mol.spin = 0 #2*S; multiplicity-1
mol.build()

# Perform RHF calculation
mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()
Energy1 = mf.hf_energy

# Redo with Newton-Raphson --> start from 'minao' guess
mf = rhf_newtonraphson.solve( mf, safe_guess=True )
Energy2 = mf.hf_energy

assert( abs( Energy1 - Energy2 ) < 1e-9 )


