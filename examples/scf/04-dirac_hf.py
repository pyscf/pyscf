#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Dirac-Hartree-Fock without time-reversal Kramers pair
'''

from pyscf import gto, scf, lib

mol = gto.M(
    atom = '''
Cl 0  0     0
H  0  1.9   0''',
    basis = 'ccpvdz',
)
mf = scf.DHF(mol)
mf.kernel()

#
# Uncontract basis of Cl, keep basis of H contracted
#
mol = gto.M(
    atom = '''
Cl 0  0     0
H  0  1.9   0''',
    basis = {'Cl': 'unc-ccpvdz',
             'H' : 'ccpvdz'},
)
lib.param.LIGHT_SPEED = 90.  # Change light speed globally

mf = scf.DHF(mol)
mf.kernel()

