#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Control the SCF procedure by different initial guess.

Spherical symmetry needs to be carefully treated in the atomic calculation.
The default initial guess may break the spherical symmetry.  Proper initial
guess can help to overcome the symmetry broken issue. It is often needed when
computing the open-shell atomic HF ground state.

See also 32-v_atom_rohf.py
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
mf.kernel()
#
# use cation to produce initial guess
#
dm1 = mf.make_rdm1()

mol.charge = 0
mol.spin = 6
mol.build(False,False)

mf = scf.RHF(mol)
mf.chkfile = 'cr_atom.chk'
# 'Ag':(6,3) means to assign 6 alpha electrons and 3 beta electrons to irrep Ag
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
mf.kernel(dm0=dm1)
# The output of .analyze() method can help to identify whether the spherical
# symmetry is conserved.
#mf.analyze()


#
# Use the converged small-basis ROHF to produce initial guess for large basis
#
mol.basis = 'aug-cc-pvdz'
mol.build(False, False)
mf = scf.RHF(mol)
mf.level_shift = .2
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
# init guess can be read from chkfile
dm = mf.from_chk('cr_atom.chk')
mf.kernel(dm)


#
# Another choice to conduct a large basis calculation from small basis resutls
# is to use second order SCF solver (.newton method).  Based on the initial
# guess from the small basis calculation which has proper spherical symmetry,
# SOSCF solver often provides reliable results that reserve the spherical
# symmetry.
#
mf1 = scf.RHF(mol).newton()
dm = mf1.from_chk('cr_atom.chk')
mf1.kernel(dm)


#
# UHF is another way to produce initial guess
#
charge = 0
spin = 6
mol.basis = 'aug-cc-pvdz'
mol.build(False,False)

mf = scf.UHF(mol)
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
mf.kernel()
dm1 = mf.make_rdm1()

mf = scf.ROHF(mol)
mf.irrep_nelec = {'Ag': (6,3), 'B1g': (1,0), 'B2g': (1,0), 'B3g': (1,0)}
mf.kernel(dm1)


#
# The third way to force the calculation strictly following the correct
# configurations is the second order SCF optimizaiton.  In the following code,
# we call a calculation on cation for a correct HF configuration with spherical
# symmetry.  This HF configuration is next pass to second order SCF solver
# (.newton method) to solve X2C-ROHF model of the open shell atom.
#
mol = gto.M(
    verbose = 4,
    symmetry = True,
    atom = [['Cr',(0, 0, 0)], ],
    basis = 'cc-pvdz-dk',
    charge = 6,
    spin = 0)
mf = scf.RHF(mol).x2c().run()
mo, mo_occ = mf.mo_coeff, mf.mo_occ

mol.charge = 0
mol.spin = 6
mol.build(False,False)

mf = scf.RHF(mol).x2c().newton()
mo_occ[9:15] = 1
mf.kernel(mo, mo_occ)
#mf.analyze()

