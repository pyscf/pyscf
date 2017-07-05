#!/usr/bin/env python
#
# Authors:  Sandeep Sharma <sanshar@gmail.com>
#           James Smith <james.smith9113@gmail.com>
#
'''
All output is deleted after the run to keep the directory neat. Comment out the
cleanup section to view output.
'''
import time; t0 = time.time()
import numpy
import math
import os
from pyscf import gto, scf, ao2mo, mcscf, tools, fci
from pyscf.future.shciscf import shci, settings



alpha = 0.007297351

mol = gto.M(
    atom = 'C 0 0 0; C 0 0 1.3119',
    basis = 'cc-pvqz',
    verbose=5,
    symmetry=1,
    spin = 2)
myhf = scf.RHF(mol)
myhf.kernel()


##USE SHCISCF
solver1 = shci.SHCI(mol)
solver1.irrep_nelec = {'A1g' : (2,1), 'A1u' :(1,1), 'E1ux' : (1,1), 'E1uy' : (1,0)}
solver1.prefix = "solver1"
solver1.epsilon2 = 1.e-7
solver1.stochastic = False

solver2 = shci.SHCI(mol)
solver2.irrep_nelec = {'A1g' : (2,1), 'A1u' :(1,1), 'E1ux' : (1,0), 'E1uy' : (1,1)}
solver2.prefix = "solver2"
solver2.epsilon2 = 1.e-7
solver2.stochastic = False

mycas = shci.SHCISCF(myhf, 8, 8)
mcscf.state_average_mix_(mycas, [solver1, solver2], numpy.ones(2)/2)
mycas.kernel()

print "Total Time:    ", time.time() - t0

# File cleanup
os.system("rm *.bkp")
os.system("rm *.txt")
os.system("rm shci.e")
os.system("rm *.dat")
os.system("rm FCIDUMP")
