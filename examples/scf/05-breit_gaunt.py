#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Enable Breit Gaunt interaction for Dirac-Hartree-Fock solver
'''

from pyscf import gto, scf

mol = gto.M(
    verbose = 5,
    atom = '''
Cl 0  0     0
H  0  1.9   0''',
    # The prefix 'unc-" decontracts basis to primitive Gaussian basis
    basis = {'Cl': 'unc-ccpvdz',
             'H' : 'ccpvdz'},
)
mf = scf.DHF(mol)
print('E(Dirac-Coulomb) = %.15g, ref = -461.443188093533' % mf.kernel())

mf.with_gaunt = True
print('E(Dirac-Coulomb-Gaunt) = %.15g, ref = -461.326149787363' % mf.kernel())

mf.with_breit = True
print('E(Dirac-Coulomb-Breit) = %.15g, ref = -461.334922770344' % mf.kernel())
