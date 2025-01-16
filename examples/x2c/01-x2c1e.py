#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Constructing the X matrix using one-electron approximation. This is the default
configuration of the X2C module.
'''

import pyscf
from pyscf.x2c import UHF as X2C_UHF
from pyscf.x2c import dft as x2c_dft

mol = pyscf.M(
    verbose = 0,
    atom = [["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
    basis = 'ccpvdz-dk',
)

method = mol.RHF()
enr = method.kernel()
print('E(NR) = %.12g' % enr)

method = X2C_UHF(mol)
ex2c = method.kernel()
print('E(X2C1E-UHF) = %.12g' % ex2c)

# Customizing basis sets for X matrix
method.with_x2c.basis = {'O': 'unc-ccpvqz', 'H':'unc-ccpvdz'}
print('E(X2C1E-UHF) = %.12g' % method.kernel())

# Constructing X matrix using single center approximation
method.with_x2c.approx = 'atom1e'
print('E(X2C1E-UHF) = %.12g' % method.kernel())

method = x2c_dft.UKS(mol).run()
print('E(X2C1E-UKS) = %.12g' % method.e_tot)
