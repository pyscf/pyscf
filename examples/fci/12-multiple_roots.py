#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf, fci

'''
Multiple FCI roots

Call fix_spin_ function here to ensure the solutions are all singlets
'''

mol = gto.M(atom='Ne 0 0 0', basis='631g')
m = scf.RHF(mol)
m.kernel()
norb = m.mo_energy.size
nelec = mol.nelectron

fs = fci.addons.fix_spin_(fci.FCI(mol, m.mo_coeff), .5)
fs.nroots = 3
e, c = fs.kernel(verbose=5)
for i, x in enumerate(c):
    print('state %d, E = %.12f  2S+1 = %.7f' %
          (i, e[i], fci.spin_op.spin_square0(x, norb, nelec)[1]))
