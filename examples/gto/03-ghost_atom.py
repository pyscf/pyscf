#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto

'''
Ghost atom has nuclear charge 0

The global basis set assignment such as ``basis = "sto3g"`` cannot be used on
ghost atom.  One needs explicitly assign basis for ghost atom using eg
:func:`gto.basis.load`.
'''

mol = gto.M(
    atom = 'C 0 0 0; ghost 0 0 2',
    basis = {'C': 'sto3g', 'ghost': gto.basis.load('sto3g', 'H')}
)


