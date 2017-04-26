#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Control the SCF procedure by different initial guess.
'''

import numpy
from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.build(
    verbose = 5,
    output = None,
    symmetry = 'D2h',
    atom = [['Cr',(0, 0, 0)], ],
    basis = 'cc-pvdz',
    charge = 6,
    spin = 0,
)
mf = scf.RHF(mol)
mf.scf()
#
# use cation to produce initial guess
#
mo = mf.mo_coeff
rdm1 = (numpy.dot(mo[:,:15], mo[:,:15].T),
        numpy.dot(mo[:,:9 ], mo[:,:9 ].T))

mol.charge = 0
mol.spin = 6
mol.build(False,False)

mf = scf.RHF(mol)
mf.chkfile = 'cr_atom.chk'
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
mf.scf(dm0=rdm1)


#
# the converged ROHF of small basis to produce initial guess for large basis
#
mol.basis = 'aug-cc-pvdz'
mol.build(False, False)
mf = scf.RHF(mol)
mf.level_shift = .2
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
# init guess can also be read from chkfile
dm = mf.from_chk('cr_atom.chk')
mf.scf(dm)


#
# UHF is another way to produce initial guess
#
charge = 0
spin = 6
mol.basis = 'aug-cc-pvdz'
mol.build(False,False)

mf = scf.UHF(mol)
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
mf.scf()
rdm1 = mf.make_rdm1()

mf = scf.RHF(mol)
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
mf.scf(rdm1)


#
# The third way to force the calculation strictly following the correct
# configurations is the second order SCF optimizaiton.  In the following code,
# we call a calculation on cation for a correct HF configuration with spherical
# symmetry.  This HF configuration is next pass to second order SCF solver
# scf.newton to solve X2C-ROHF model of the open shell atom.
#
mol = gto.M(
    verbose = 4,
    symmetry = True,
    atom = [['Cr',(0, 0, 0)], ],
    basis = 'cc-pvdz-dk',
    charge = 6,
    spin = 0)
mf = scf.sfx2c1e(scf.RHF(mol)).run()
mo, mo_occ = mf.mo_coeff, mf.mo_occ

mol.charge = 0
mol.spin = 6
mol.build(False,False)

mf = scf.newton(scf.sfx2c1e(scf.RHF(mol)))
mo_occ[9:15] = 1
mf.kernel(mo, mo_occ)
#mf.analyze()

