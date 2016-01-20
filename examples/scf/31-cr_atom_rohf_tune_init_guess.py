#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto
from pyscf import scf

'''
Control the SCF procedure by different initial guess.
'''

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
m = scf.RHF(mol)
m.scf()
#
# use cation to produce initial guess
#
mo = m.mo_coeff
rdm1 = (numpy.dot(mo[:,:15], mo[:,:15].T),
        numpy.dot(mo[:,:9 ], mo[:,:9 ].T))

mol.charge = 0
mol.spin = 6
mol.build(False,False)

m = scf.RHF(mol)
m.chkfile = 'cr_atom.chk'
m.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
m.scf(dm0=rdm1)


#
# the converged ROHF of small basis to produce initial guess for large basis
#
mol.basis = 'aug-cc-pvdz'
mol.build(False, False)
m = scf.RHF(mol)
m.level_shift = .2
m.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
# init guess can also be read from chkfile
dm = m.from_chk('cr_atom.chk')
m.scf(dm)


#
# UHF is another way to produce initial guess
#
charge = 0
spin = 6
mol.basis = 'aug-cc-pvdz'
mol.build(False,False)

m = scf.UHF(mol)
m.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
m.scf()
rdm1 = m.make_rdm1()

m = scf.RHF(mol)
m.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
m.scf(rdm1)
