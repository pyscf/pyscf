#!/usr/bin/env python

'''
Mulliken population analysis with NAO
'''

import numpy
from pyscf import gto, scf, lo
from functools import reduce

x = .63
mol = gto.M(atom=[['C', (0, 0, 0)],
                  ['H', (x ,  x,  x)],
                  ['H', (-x, -x,  x)],
                  ['H', (-x,  x, -x)],
                  ['H', ( x, -x, -x)]],
            basis='ccpvtz')
mf = scf.RHF(mol).run()

# C matrix stores the AO to localized orbital coefficients
C = lo.orth_ao(mf, 'nao')

# C is orthogonal wrt to the AO overlap matrix.  C^T S C  is an identity matrix.
print(abs(reduce(numpy.dot, (C.T, mf.get_ovlp(), C)) -
          numpy.eye(mol.nao_nr())).max())  # should be close to 0

# The following linear equation can also be solved using the matrix
# multiplication reduce(numpy.dot (C.T, mf.get_ovlp(), mf.mo_coeff))
mo = numpy.linalg.solve(C, mf.mo_coeff)

#
# Mulliken population analysis based on NAOs
#
dm = mf.make_rdm1(mo, mf.mo_occ)
mf.mulliken_pop(mol, dm, numpy.eye(mol.nao_nr()))

# For transition metals, the core/valence/Rydberg partition used to construct
# NAOs can be adjusted before calling lo.orth_ao or lo.nao.nao.
# These settings change the global configuration for the element. Successive
# calls retain unspecified shell counts, so the following changes accumulate.
# String descriptions leave the core unchanged and set the named valence shells:
# lo.nao.set_atom_conf('Fe', '1s1d')      # Set one s and one d valence shell.
# lo.nao.set_atom_conf('Fe', 'double d')  # Set two d valence shells.
# lo.nao.set_atom_conf('Fe', 'double p')  # Set two p valence shells.
# lo.nao.set_atom_conf('Fe', 'polarize')  # Add the first absent angular momentum
#                                      # shell (an f shell in this sequence).
