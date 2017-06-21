#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to control DIIS parameters and how to use different
DIIS schemes (CDIIS, ADIIS, EDIIS) in SCF calculations.

Note the calculations in this example is served as a demonstration for the use
of DIIS.  Without other convergence technique, none of them can converge.
'''

from pyscf import gto, scf, dft

mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*')

#
# Default DIIS scheme is CDIIS.  DIIS parameters can be assigned to mf object
# with prefix ".diis_"
#
mf = scf.RKS(mol)
mf.diis_space = 14
mf.kernel()

#
# mf.diis can be replaced by other DIIS objects (ADIIS, EDIIS).
# In this case, DIIS parameters specified in the SCF object (mf.diis_space,
# mf.diis_file, ...) have no effects.  DIIS parameters need to be assigned to
# the given diis object.
#
my_diis_obj = scf.ADIIS()
my_diis_obj.diis_space = 12
mf.diis = my_diis_obj
mf.kernel()

my_diis_obj = scf.EDIIS()
my_diis_obj.diis_space = 12
mf.diis = my_diis_obj
mf.kernel()

