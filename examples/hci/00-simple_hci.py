#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run heat-bath selected CI
'''

from pyscf import gto, scf, ao2mo, hci

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = '6-31g',
)
myhf = scf.RHF(mol).run()

cisolver = hci.SCI(mol)
nmo = myhf.mo_coeff.shape[1]
nelec = mol.nelec
h1 = myhf.mo_coeff.T.dot(myhf.get_hcore()).dot(myhf.mo_coeff)
h2 = ao2mo.full(mol, myhf.mo_coeff)
e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)

