#!/usr/bin/env python

'''
Mulliken population analysis with NAO
'''

import numpy
from pyscf import gto, scf, lo
from pyscf.lo.nao import _prenao_sub, _core_val_ryd_list
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

# The pre-NAO step diagonalizes the atomic density blocks. Its occupations
# and the final NAO occupations expose how density is distributed by shell.
s = mf.get_ovlp()
p = reduce(numpy.dot, (s, mf.make_rdm1(), s))
pre_nao_occ, _ = _prenao_sub(mol, p, s)
print('Pre-NAO occupations:', pre_nao_occ)
print('NAO occupations:', reduce(numpy.dot, (C.T, p, C)).diagonal())
print('Core, valence and Rydberg shells:', _core_val_ryd_list(mol))

# For transition metals, the core/valence/Rydberg partition used to construct
# NAOs can be adjusted before calling lo.orth_ao or lo.nao.nao. For example,
# include a second d shell in the Fe valence space:
# lo.nao.set_atom_conf('Fe', 'double d')
