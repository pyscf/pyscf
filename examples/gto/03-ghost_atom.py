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
# Specify diffierent basis for different ghost atoms
#
mol = gto.M(atom='''
ghost1     0.000000000     0.000000000     2.500000000
ghost2    -0.663641000    -0.383071000     3.095377000
ghost2     0.663588000     0.383072000     3.095377000
O     1.000000000     0.000000000     2.500000000
H    -1.663641000    -0.383071000     3.095377000
H     1.663588000     0.383072000     3.095377000
''',
            basis={'ghost1':gto.basis.load('sto3g', 'O'),
                   'ghost2':gto.basis.load('631g', 'H'),
                   'O':'631g', 'H':'631g'}
)


#
# Add "ghost" as prefix for the ghost atom.  In this input, the atoms prefixed
# with ghost are ghost atoms.  Their charges are zero.  However, the basis set
# of the unmodified atom is applied for the ghost atom.  In the following
# example, ghost-O uses the 6-31G basis of oxygen atom.
#
mol = gto.M(atom='''
ghost-O     0.000000000     0.000000000     2.500000000
ghost_H    -0.663641000    -0.383071000     3.095377000
ghost:H     0.663588000     0.383072000     3.095377000
O     1.000000000     0.000000000     2.500000000
H    -1.663641000    -0.383071000     3.095377000
H     1.663588000     0.383072000     3.095377000
''', basis='631g')

#
# "X" can also be used as the label for ghost atoms
#
mol = gto.M(atom='''
X1     0.000000000     0.000000000     2.500000000
X2    -0.663641000    -0.383071000     3.095377000
X2     0.663588000     0.383072000     3.095377000
O     1.000000000     0.000000000     2.500000000
H    -1.663641000    -0.383071000     3.095377000
H     1.663588000     0.383072000     3.095377000
''',
            basis={'X1':gto.basis.load('sto3g', 'O'),
                   'X2':gto.basis.load('631g', 'H'),
                   'O':'631g', 'H':'631g'}
)

mol = gto.M(atom='''
X-O     0.000000000     0.000000000     2.500000000
X_H    -0.663641000    -0.383071000     3.095377000
X:H     0.663588000     0.383072000     3.095377000
O     1.000000000     0.000000000     2.500000000
H    -1.663641000    -0.383071000     3.095377000
H     1.663588000     0.383072000     3.095377000
''', basis='631g')
