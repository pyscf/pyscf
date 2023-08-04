#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Multipole integrals.  (Be careful with the gauge origin of the multipole
integrals).  An implementation of multipoles (up to hexadecapole) can be found
in
https://github.com/cuanto/pyscf-scripts/blob/master/props/multipole_rhf.py

See also 20-ao_integrals for more examples to access integrals
'''

import numpy
from pyscf import gto, scf

mol = gto.M(
    verbose = 0,
    atom = 'H 0 0 0; H 0 0 1.5; H 0 1 1; H 1.1 0.2 0',
    basis = 'ccpvdz'
)

nao = mol.nao

# Dipole integral
dip = mol.intor('int1e_r').reshape(3,nao,nao)

# Quadrupole
quad = mol.intor('int1e_rr').reshape(3,3,nao,nao)

# Octupole
octa = mol.intor('int1e_rrr').reshape(3,3,3,nao,nao)

# hexadecapole
hexa = mol.intor('int1e_rrrr').reshape(3,3,3,3,nao,nao)


