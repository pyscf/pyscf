#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: September 2, 2015
#
# Test file to illustrate the augmented Hessian Newton-Raphson RHF routine
#

from pyscf import gto, scf
from pyscf.tools import molden
from pyscf.tools import rhf_newtonraphson
from pyscf.lib import parameters as param
import numpy as np

mol = gto.Mole() # Caffeine optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
    C            1.817879727385     2.153638906169     0.000000000000
    C           -0.346505640687     3.462095703243     0.000000000000
    N            0.463691886339     2.246640891586     0.000000000000
    N            2.269654714050     0.899687380469     0.000000000000
    C           -0.003923667257     0.937835225180     0.000000000000
    C            1.134487063408     0.149351644916     0.000000000000
    C           -1.338251344864     0.403980438637     0.000000000000
    O           -2.385251021476     1.048239461893     0.000000000000
    N            1.058400655395    -1.225474274555     0.000000000000
    C            2.251397069015    -2.069224507359     0.000000000000
    C           -0.187094195596    -1.846487404954     0.000000000000
    O           -0.292432240842    -3.063478138223     0.000000000000
    N           -1.323645520078    -1.013034361391     0.000000000000
    C           -2.609849686479    -1.716498251901     0.000000000000
    H           -2.689209473523    -2.357766508547     0.889337124082
    H           -2.689209473523    -2.357766508547    -0.889337124082
    H           -3.397445032059    -0.956544015308     0.000000000000
    H            3.126343795339    -1.409688574359     0.000000000000
    H            2.260091440174    -2.714879611857    -0.890143668130
    H            2.260091440174    -2.714879611857     0.890143668130
    H            2.453380552599     3.037434448146     0.000000000000
    H           -1.400292735506     3.159575123448     0.000000000000
    H           -0.135202960256     4.062674697502     0.897532201407
    H           -0.135202960256     4.062674697502    -0.897532201407
  '''
mol.basis = '6-31g'
mol.symmetry = 1
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

mf = scf.RHF( mol )
mf.verbose = 4
mf.max_cycle = 1
mf.scf()
DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
if ( mf.converged == False ):
    mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )



