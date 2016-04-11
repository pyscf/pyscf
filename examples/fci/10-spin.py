#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Assign spin state for FCI wavefunction.

By default, the FCI solver will take Mole attribute spin for the spin state.
It can be overwritten by passing kwarg ``nelec`` to the kernel function of FCI
solver.  The nelec argument is a two-element tuple.  The first is the number
of alpha electrons; the second is the number of beta electrons.

If spin-contamination is observed on FCI wavefunction, we can use the
decoration function :func:`fci.addons.fix_spin_` to level shift the energy of
states which do not have the target spin.
'''

import numpy
from pyscf import gto, scf, fci

mol = gto.M(atom='Ne 0 0 0', basis='631g', spin=2)
m = scf.RHF(mol)
m.kernel()
norb = m.mo_energy.size

fs = fci.FCI(mol, m.mo_coeff)
e, c = fs.kernel()
print('E = %.12f  2S+1 = %.7f' %
      (e, fci.spin_op.spin_square0(c, norb, (6,4))[1]))

e, c = fs.kernel(nelec=(5,5))
print('E = %.12f  2S+1 = %.7f' %
      (e, fci.spin_op.spin_square0(c, norb, (5,5))[1]))


fs = fci.addons.fix_spin_(fci.FCI(mol, m.mo_coeff), shift=.5)
e, c = fs.kernel()
print('E = %.12f  2S+1 = %.7f' %
      (e, fci.spin_op.spin_square0(c, norb, (6,4))[1]))
