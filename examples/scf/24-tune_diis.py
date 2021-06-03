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
mf.diis_file = 'o2_diis.h5'
mf.run()

#
# We can use other DIIS methods (ADIIS, EDIIS) in SCF. There are multiple
# approaches to set DIIS methods.
#
#   1. Set DIIS method with the attribute mf.DIIS. The value of mf.DIIS needs
#   to be a subclass of lib.diis.DIIS class. scf module provides three DIIS
#   classes: CDIIS, ADIIS, EDIIS
#
mf.DIIS = scf.ADIIS
mf.diis_space = 14
mf.run()

mf.DIIS = scf.EDIIS
mf.diis_space = 14
mf.run()

#
#   2. Overwrite the attribute mf.diis.  In this approach, DIIS parameters
#   specified in the SCF object (mf.diis_space, mf.diis_file, ...) have no
#   effects.  DIIS parameters need to be assigned to the diis object.
#
my_diis_obj = scf.ADIIS()
my_diis_obj.space = 12
my_diis_obj.filename = 'o2_diis.h5'
mf.diis = my_diis_obj
mf.run()

my_diis_obj = scf.EDIIS()
my_diis_obj.space = 12
mf.diis = my_diis_obj
mf.run()

#
# By creating an DIIS object and assigning it to the attribute mf.diis, we can
# restore SCF iterations from an existed SCF calculation (see also the example
# 14-restart.py)
#
mf = mol.RKS()
mf.diis = scf.ADIIS().restore('h2o_diis.h5')
mf.run()
