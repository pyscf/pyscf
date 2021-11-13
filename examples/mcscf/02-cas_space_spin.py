#!/usr/bin/env python

'''
spin settings in active space

The mol.spin attribute controls the Sz value of the molecule. CASCI/CASSCF
methods by default use this parameter to determine Sz of the correlated
wave-function in active space (i.e. the number of alpha and beta electrons).
The number of alpha and beta electrons can be set independently in the mcscf
methods.
'''

import pyscf

mol = pyscf.M(
    atom = 'C 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 0)

myhf = mol.RHF().run()

# 6 orbitals, 6 electrons. 3 alpha electrons and 3 beta electrons will be
# assigned to the active space due to mol.spin = 0
# This setting tends to converge to the singlet state.
mycas = myhf.CASSCF(6, 6).run()

# 6 orbitals, 4 alpha electrons, 2 beta electrons.
# This setting tends to converge to the triplet state
mycas = myhf.CASSCF(6, (4, 2)).run()

# 6 orbitals, 3 alpha electrons, 3 beta electrons, but solving the quintet
# state. See also example 18-spatial_spin_symmetry.py
mycas = myhf.CASSCF(6, (3, 3))
mycas.fix_spin_(ss=6)
mycas.run()
