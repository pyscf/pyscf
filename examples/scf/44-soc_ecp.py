#!/usr/bin/env python

'''
SCF module currently does not apply SO-ECP automatically. SO-ECP contributions
can be added to GHF/GKS core Hamiltonian by overwriding the method get_hcore.
Since pyscf-2.0 setting attribte with_soc in GHF object can include the
ECP-SOC integrals in core Hamiltonian.

See also examples/gto/20-soc_ecp.py
'''

import numpy
from pyscf import gto, lib

mol = gto.M(
    verbose = 4,
    atom = 'C 0 0 0; O 0 0 1.5',
    basis = {'C': 'crenbl', 'O': 'ccpvdz'},
    ecp = {'C': 'crenbl'}
)

#
# Adding ECP-SOC contribution to GHF Hamiltonian
#
mf = mol.GHF()
s = .5 * lib.PauliMatrices
# ECPso evaluates SO-ECP integrals
#       <i| 1j * l U(r)|j>
# Note to the phase factor -1j to remove the phase 1j above when adding to
# core Hamiltonian
ecpso = -1j * lib.einsum('sxy,spq->xpyq', s, mol.intor('ECPso'))
hcore = mf.get_hcore()
hcore = hcore + ecpso.reshape(hcore.shape)
mf.get_hcore = lambda *args: hcore
mf.kernel()

#
# Since pyscf-2.0 ECP-SOC can be enabled in GHF object
#
mf = mol.GHF()
mf.with_soc = True
mf.kernel()
