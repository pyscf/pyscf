#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Assign spin state for HCI wavefunction.

If spin-contamination is observed for HCI wavefunction, :func:`hci.fix_spin`
function can be used to level shift the states of wrong spin.  This is often
helpful to reduce the spin-contamination.
'''

from pyscf import gto, scf, ao2mo
from pyscf.hci import hci

mol = gto.M(atom='O 0 0 0', basis='631g', spin=0)
myhf = scf.RHF(mol).run()

cisolver = hci.SCI(mol)
nmo = myhf.mo_coeff.shape[1]
nelec = mol.nelec
h1 = myhf.mo_coeff.T.dot(myhf.get_hcore()).dot(myhf.mo_coeff)
h2 = ao2mo.full(mol, myhf.mo_coeff)
e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)
print('E = %.12f  2S+1 = %.7f' %
      (e, cisolver.spin_square(civec[0], nmo, nelec)[1]))

cisolver = hci.fix_spin(cisolver, ss=0)  # ss = S^2
e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)
print('E = %.12f  2S+1 = %.7f' %
      (e, cisolver.spin_square(civec[0], nmo, nelec)[1]))
