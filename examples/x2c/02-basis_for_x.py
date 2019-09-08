#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
X2c method use the uncontracted large component basis to construct the X
matrix. The basis for X matrix can have a big impact to the total energy.
X2c treatment is not variational. The total energy often increases when the
quality of large component basis is improved. This is due to the fact that the
X2c ground state is an "excited" state in the framework of full-relativistic
spectrum. When X2c Hamiltonian was constructed, the negative states are
projected out. Excited state is not bounded from below. Depending on the
quality of the basis for large component (which dominates positive states) and
small component (which dominates negative states), X2c energy may first
decrease then increase when systematically increase the size of basis set.

This example shows how to adjust X basis so that you get consistent
contributions from X matrix for different large component basis. When the
uncertainty of X matrix was removed, you may observe the monotonic energy
convergence wrt basis size.
'''

from pyscf import gto
from pyscf import scf

# A combined basis: uncontracted ANO basis plus some steep even-tempered Gaussians
xbasis = ('unc-ano',
          gto.etbs([(0, 8, 1e7, 2.5),   # s-function
                    (1, 5, 5e4, 2.5),   # p-function
                    (2, 2, 1e3, 2.5)])) # d-function

mol = gto.M(atom = 'Zn',
    basis = 'ccpvdz-dk',
)
mf = scf.RHF(mol).x2c()
# Assigning a different basis to X matrix
mf.with_x2c.basis = xbasis
mf.run()

mol = gto.M(atom = 'Zn',
    basis = 'ccpvtz-dk',
)
mf = scf.RHF(mol).x2c()
mf.with_x2c.basis = xbasis
mf.run()

mol = gto.M(atom = 'Zn',
    basis = 'ccpvqz-dk',
)
mf = scf.RHF(mol).x2c()
mf.with_x2c.basis = xbasis
mf.run()
