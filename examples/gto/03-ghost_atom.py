#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Ghost atom has nuclear charge 0

The global basis set assignment such as ``basis = "sto3g"`` cannot be used on
ghost atom.  One needs explicitly assign basis for ghost atom using eg
:func:`gto.basis.load`.
'''

from pyscf import gto

mol = gto.M(
    atom = 'C 0 0 0; ghost 0 0 2',
    basis = {'C': 'sto3g', 'ghost': gto.basis.load('sto3g', 'H')}
)

#
# Add "ghost" as prefix for the ghost atom.  In this input, the basis set of the
# unmodified atom is applied for the ghost atom.  In the following example,
# ghost-O uses the O 6-31G basis.
#
mol = gto.M(atom='''
ghost-O     0.000000000     0.000000000     2.500000000
ghost_H    -0.663641000    -0.383071000     3.095377000
ghost:H     0.663588000     0.383072000     3.095377000
O     1.000000000     0.000000000     2.500000000
H    -1.663641000    -0.383071000     3.095377000
H     1.663588000     0.383072000     3.095377000
''', basis='631g')

