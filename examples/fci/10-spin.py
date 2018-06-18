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
      (e, fs.spin_square(c, norb, (6,4))[1]))

e, c = fs.kernel(nelec=(5,5))
print('E = %.12f  2S+1 = %.7f' %
      (e, fs.spin_square(c, norb, (5,5))[1]))


fs = fci.addons.fix_spin_(fci.FCI(mol, m.mo_coeff), shift=.5)
e, c = fs.kernel()
print('E = %.12f  2S+1 = %.7f' %
      (e, fs.spin_square(c, norb, (6,4))[1]))


#
# Example 2:  Oxygen molecule singlet state
#

nelec = (8,8)
mol = gto.M(atom='O 0 0 0; O 0 0 1.2', spin=2, basis='sto3g',
            symmetry=1, verbose=0)
mf = scf.RHF(mol).run()
mci = fci.FCI(mol, mf.mo_coeff)
mci.wfnsym = 'A1g'
mci = fci.addons.fix_spin_(mci, ss=0)
e, civec = mci.kernel(nelec=nelec)
print('A1g singlet E = %.12f  2S+1 = %.7f' %
      (e, mci.spin_square(civec, mf.mo_coeff.shape[1], nelec)[1]))

mci.wfnsym = 'A2g'
mci = fci.addons.fix_spin_(mci, ss=0)
e, civec = mci.kernel(nelec=nelec)
print('A2g singlet E = %.12f  2S+1 = %.7f' %
      (e, mci.spin_square(civec, mf.mo_coeff.shape[1], nelec)[1]))

mol = gto.M(atom='O 0 0 0; O 0 0 1.2', spin=2, basis='sto3g',
            verbose=0)
mf = scf.RHF(mol).run()
mci = fci.FCI(mol, mf.mo_coeff)
mci = fci.addons.fix_spin_(mci, ss=0)
e, civec = mci.kernel(nelec=nelec)
print('Singlet E = %.12f  2S+1 = %.7f' %
      (e, mci.spin_square(civec, mf.mo_coeff.shape[1], nelec)[1]))
