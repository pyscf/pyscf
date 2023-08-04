#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate an active space Hamiltonian from mcscf object
'''

import numpy
from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'ccpvdz')
myhf = scf.RHF(mol)
myhf.kernel()
#myhf.verbose = 5
#myhf.analyze()

#
# Generating the active space Hamiltonian with get_h1eff and get_h2eff
# function
#
norb = 6
nelec = 6
mycas = mcscf.CASSCF(myhf, norb, nelec)
h1e_cas, ecore = mycas.get_h1eff()
h2e_cas = mycas.get_h2eff()

#
# The active space Hamiltonian can be used in a FCI calculation.
#
from pyscf import fci
e, fcivec = fci.direct_spin1.kernel(h1e_cas, h2e_cas, norb, nelec,
                                    ecore=ecore, verbose=5)
print('Total energy', e)

