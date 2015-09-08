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
mf.irrep_nelec = {'A1g': 4, 'E1gx': 2, 'E1gy': 2}
e = mf.kernel()
print('E = %.15g  ref = -74.1112374269129' % e)


mol.symmetry = 'D2h'
mol.charge = 1
mol.spin = 1
mol.build(dump_input=False, parse_arg=False)
mf = scf.RHF(mol)
mf.irrep_nelec = {'Ag': (2,1), 'B1u': 4, 'B2u': 2, 'B3u': 2,}
e = mf.kernel()
print('E = %.15g  ref = -74.4026583773135' % e)


mf.irrep_nelec = {'Ag': 4, 'B1u': (2,1), 'B2u': 2, 'B3u': 2,}
e = mf.kernel()
print('E = %.15g  ref = -74.8971476600812' % e)
