#!/usr/bin/env python

'''
SCF module currently does not apply SO-ECP automatically. SO-ECP contributions
can be added to GHF/GKS core Hamiltonian by overwriding the method get_hcore.

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

mf = mol.GHF()
s = .5 * lib.PauliMatrices
ecpso = lib.einsum('sxy,spq->pxqy', s, mol.intor('ECPso'))
hcore = mf.get_hcore()
hcore = hcore + ecpso.reshape(hcore.shape)
mf.get_hcore = lambda *args: hcore
mf.kernel()
