#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf

'''
Specify irrep_nelec to control the wave function symmetry
'''


mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''
        C     0.   0.   0.625
        C     0.   0.  -0.625 ''',
    basis = 'cc-pVDZ',
    spin = 0,
    symmetry = True,
)

mf = scf.RHF(mol)

# Frozen occupancy
# 'A1g': 4 electrons
# 'E1gx': 2 electrons
# 'E1gy': 2 electrons
# Rest 4 electrons are put in irreps A1u, E1ux, E1uy ... based on Aufbau principle
# The irrep names can be found in pyscf/symm/param.py
mf.irrep_nelec = {'A1g': 4, 'E1gx': 2, 'E1gy': 2}
e = mf.kernel()
print('E = %.15g  ref = -74.1112374269129' % e)


mol.symmetry = 'D2h'
mol.charge = 1
mol.spin = 1
mol.build(dump_input=False, parse_arg=False)
mf = scf.RHF(mol)

# Frozen occupancy
# 'Ag': 2 alpha, 1 beta electrons
# 'B1u': 4 electrons
# 'B2u': 2 electrons
# 'B3u': 2 electrons
mf.irrep_nelec = {'Ag': (2,1), 'B1u': 4, 'B2u': 2, 'B3u': 2,}
e = mf.kernel()
print('E = %.15g  ref = -74.4026583773135' % e)

# Frozen occupancy
# 'Ag': 4 electrons
# 'B1u': 2 alpha, 1 beta electrons
# 'B2u': 2 electrons
# 'B3u': 2 electrons
mf.irrep_nelec = {'Ag': 4, 'B1u': (2,1), 'B2u': 2, 'B3u': 2,}
e = mf.kernel()
print('E = %.15g  ref = -74.8971476600812' % e)
