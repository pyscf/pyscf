#!/usr/bin/env python

import numpy
from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.build(
    verbose = 5,
    output = None,
    symmetry = True,
    atom = [['Cr',(0, 0, 0)], ],
    basis = {'Cr': 'cc-pvdz', },
    charge = 6,
    spin = 0,
)
m = scf.RHF(mol)
m.scf()
mo = m.mo_coeff

#
# use cation to produce first initial guess
#
rdm1 = (numpy.dot(mo[:,:15], mo[:,:15].T),
        numpy.dot(mo[:,:9 ], mo[:,:9 ].T))


mol.charge = 0
mol.spin = 6
mol.build(False,False)

m = scf.RHF(mol)
m.chkfile = 'cr_atom.chk'
m.irrep_nocc_alpha = {'Ag': 6, 'B1g': 1, 'B2g': 1, 'B3g': 1,}
m.irrep_nocc_beta  = {'Ag': 3, 'B1g': 0, 'B2g': 0, 'B3g': 0,}
# init guess by atom or 1e cannot make ROHF converge due to symmetry broken
# during the iteration. Using the closed RHF ion to guide the ROHF works well
#rdm1 = scf.init_guess_by_atom(mol)
#rdm1 = scf.init_guess_by_1e(mol)
m.scf(dm0=rdm1)


#
# the converged ROHF of small basis to produce initial guess for large basis
#
mol.basis = 'aug-cc-pvdz'
mol.build(False, False)
m = scf.RHF(mol)
m.level_shift_factor = .2
m.irrep_nocc_alpha = {'Ag': 6, 'B1g': 1, 'B2g': 1, 'B3g': 1,}
m.irrep_nocc_beta  = {'Ag': 3, 'B1g': 0, 'B2g': 0, 'B3g': 0,}
# init guess can also be read from chkfile
m.make_init_guess = scf.hf.init_guess_by_chkfile(mol, 'cr_atom.chk')
m.scf()



#
# UHF is another way to produce initial guess
#
charge = 0
spin = 6
mol.basis = 'aug-cc-pvdz'
mol.build(False,False)

m = scf.UHF(mol)
m.irrep_nocc_alpha = {'Ag': 6, 'B1g': 1, 'B2g': 1, 'B3g': 1,}
m.irrep_nocc_beta  = {'Ag': 3, 'B1g': 0, 'B2g': 0, 'B3g': 0,}
m.scf()
rdm1 = m.make_rdm1()

m = scf.RHF(mol)
m.irrep_nocc_alpha = {'Ag': 6, 'B1g': 1, 'B2g': 1, 'B3g': 1,}
m.irrep_nocc_beta  = {'Ag': 3, 'B1g': 0, 'B2g': 0, 'B3g': 0,}
m.scf(rdm1)
