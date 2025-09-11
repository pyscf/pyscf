#!/usr/bin/env python

'''
This example covers different ways to input basis
1. A universal basis set for all elements.
2. Different basis for different elements.
3. Different basis for same elements (different atoms).
4. Default basis for all elements, except the given basis of specific element.
5. gto.basis.parse and gto.basis.load functions to input user-specified basis.
6. Reading the basis set from a given file.
7. Defining custom aliases for basis sets stored in files.
8. Uncontracted basis with prefix "unc-".
9. Basis truncation and a subset of a basis set with notation "@".
10. Even tempered gaussian basis.
11. Combining multiple basis sets into one basis set.
12. Internal format (not recommended)
'''

import os
import numpy
from pyscf import gto

dirnow = os.path.realpath(os.path.join(__file__, '..'))
basis_file_from_user = os.path.join(dirnow, 'h_sto3g.dat')

#
# Simplest input, one basis set for all atoms.
#
mol = gto.M(atom = '''O 0 0 0; H 0 1 0; H 0 0 1''',
            basis = 'ccpvdz')

#
# Different basis for different elements.
# The attribute mol.basis needs to be initialized as a dict in which the
# key is element and the value is the basis set.
#
# You can use atomic symbol (case-insensitive), the atomic nuclear charge,
# as the key of the mol.basis dict.
#
mol = gto.M(atom = '''O 0 0 0; 1 0 1 0; H 0 0 1''',
            basis = {8: 'ccpvdz', 'h': 'sto3g'})

#
# Different basis for same type of elements (different atoms).
# Tag the atoms with special characters 1234567890~!@#$%^&*()_+.?:<>[]{}|  to
# distinguish different atoms.  If the basis set of tagged atom is not
# specified in the basis dict, the default basis will be taken.
# In the following input, sto3g basis will be assigned to H:1, 631g basis will
# be assigned to H@2
#
mol = gto.M(atom = '''O 0 0 0; H:1 0 1 0; H@2 0 0 1''',
            basis = {'O': 'ccpvdz', 'H:1': 'sto3g', 'H': '631g'})

#
# Set a default basis set for all elements.  If a specific basis is assigned to
# an element, the basis will be used for the specific element.
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'default': '6-31g', 'H2': 'sto3g'}
)

#
# Use gto.basis.parse and gto.basis.load functions to input basis
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'O': gto.parse('''
# Parse NWChem format basis string (see https://www.basissetexchange.org/).
# Comment lines are ignored
#BASIS SET: (6s,3p) -> [2s,1p]
O    S
    130.7093200              0.15432897
     23.8088610              0.53532814
      6.4436083              0.44463454
O    SP
      5.0331513             -0.09996723             0.15591627
      1.1695961              0.39951283             0.60768372
      0.3803890              0.70011547             0.39195739
                                '''),
             'H1': basis_file_from_user,
             'H2': gto.load('sto-3g', 'He')  # or use basis of another atom
            }
)

#
# You can define custom aliases for basis sets defined in files.
# The variables USER_BASIS_DIR and USER_BASIS_ALIAS should be defined
# in your PySCF configuration file (e.g. ~/.pyscf_conf.py) like this:

USER_BASIS_DIR = dirnow
USER_BASIS_ALIAS = {'customsto3g' : 'h_sto3g.dat'}

# We need to override USER_BASIS_DIR and USER_BASIS_ALIAS so that the example can run.
# Don't actually use the next two lines in your code.
gto.basis.USER_BASIS_DIR = USER_BASIS_DIR
gto.basis.USER_BASIS_ALIAS = USER_BASIS_ALIAS

mol = gto.M(
    atom = '''H 0 0 0; H 0 0 1''',
    basis = 'customsto3g'
)


#
# Uncontracted basis, decontracting basis.
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'O': 'unc-ccpvdz', # prefix "unc-" will uncontract the ccpvdz basis.
                                # It is equivalent to assigning
                                #   'O': gto.uncontract(gto.load('ccpvdz', 'O')),
             'H': 'ccpvdz'  # H1 H2 will use the same basis ccpvdz
            }
)

#
# Basis truncation, basis subset with notation "@"
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'O': 'ano@3s2p',  # Truncate the ANO basis and keep only 9
                               # functions (3s, 2p) for O atom.
             'H': 'ccpvdz@1s'  # One s function from cc-pVDZ basis
            }
)


#
# Even tempered gaussian basis
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'H': 'sto3g',
# even-temper gaussians alpha*beta^i, where i = 0,..,n
#                                  (l, n, alpha, beta)
             'O': gto.etbs([(0, 4, 1.5, 2.2),  # s-function
                            (1, 2, 0.5, 2.2)]) # p-function
            }
)


#
# Internal format (not recommended). See also
# pyscf/gto/basis/dzp_dunning.py  as an example of internal format
#
# basis = {atom_type1:[[angular_momentum
#                       (GTO-exp1, contract-coeff11, contract-coeff12),
#                       (GTO-exp2, contract-coeff21, contract-coeff22),
#                       (GTO-exp3, contract-coeff31, contract-coeff32),
#                       ...],
#                      [angular_momentum
#                       (GTO-exp1, contract-coeff11, contract-coeff12),
#                       ...],
#                      ...],
#          atom_type2:[[angular_momentum, (...),],
#                      ...],
#
mol.basis = {'H': [[0,
                    (19.2406000, 0.0328280),
                    (2.8992000, 0.2312080),
                    (0.6534000, 0.8172380),],
                   [0,
                    (0.1776000, 1.0000000),],
                   [1,
                    (1.0000000, 1.0000000),]],
}
mol.build()  # You should see a warning message here since basis set for O is not specified

#
# Functions gto.basis.load and gto.basis.parse  convert the input to the
# internal format
#
mol.basis = {'H': gto.basis.load('sto3g', 'H'),
             'O': gto.basis.parse('''
C    S
     71.6168370              0.15432897
     13.0450960              0.53532814
      3.5305122              0.44463454
C    SP
      2.9412494             -0.09996723             0.15591627
      0.6834831              0.39951283             0.60768372
      0.2222899              0.70011547             0.39195739
''')}
mol.build()

#
# If a string of basis set was input, basis parser can make a guess and call
# gto.basis.parse automatically.  The following basis input is equivalent to
# the one above.
#
mol.basis = {'H': gto.basis.load('sto3g', 'H'),
             'O': '''
C    S
     71.6168370              0.15432897
     13.0450960              0.53532814
      3.5305122              0.44463454
C    SP
      2.9412494             -0.09996723             0.15591627
      0.6834831              0.39951283             0.60768372
      0.2222899              0.70011547             0.39195739
'''}
mol.build()

# Note the rule to de-contract basis also works here: If the basis string is
# prefixed with unc, the basis set will be uncontracted.
mol.basis = {'H': gto.basis.load('sto3g', 'H'),
             'O': '''unc
C    S
     71.6168370              0.15432897
     13.0450960              0.53532814
      3.5305122              0.44463454
C    SP
      2.9412494             -0.09996723             0.15591627
      0.6834831              0.39951283             0.60768372
      0.2222899              0.70011547             0.39195739
'''}
mol.build()

#
# Simple arithmetic expressions can be specified in basis input
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'O': '''
O    S
    130.7093200*np.exp(.5)              0.15432897
     23.8088610*np.exp(.5)              0.53532814
      6.4436083*np.exp(.5)              0.44463454
O    SP
      5.0331513*2**.5          -0.09996723             0.15591627
      1.1695961*2**.5           0.39951283             0.60768372
      0.3803890*2**.5           0.70011547             0.39195739''',
             'H': 'sto3g',
            }
)

#
# Multiple basis set can be combined and used as a union basis set
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = ('sto3g', 'ccpvdz', '3-21g',
             gto.etbs([(0, 4, 1.5, 2.2), (1, 2, 0.5, 2.2)]),
            [[0, numpy.array([1e3, 1.])]])
)
print('nao = ', mol.nao_nr())

#
# The combined basis sets can be assigned to specific elements.
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'default': ('6-31g', [[0, [.05, 1.]]]),
             'H2': ['sto3g', gto.basis.parse('''
C    S
     71.6168370              0.15432897
     13.0450960              0.53532814
      3.5305122              0.44463454
C    SP
      2.9412494             -0.09996723             0.15591627
      0.6834831              0.39951283             0.60768372
      0.2222899              0.70011547             0.39195739
''')]}
)
print('nao = ', mol.nao_nr())

#
# Optimize the basis contraction.  When optimize=True is specified in the
# parse function, the segment contracted basis can be restructured to
# general contracted basis.  This can improve integral performance.
#
mol = gto.M(
    atom = '''O 0 0 0; H 0 1 0; H 0 0 1''',
    basis = {'O': '631g',
             'H': gto.basis.parse('''
H    S
      2.9412494             -0.09996723
      0.6834831              0.39951283
      0.2222899              0.70011547
H    S
      2.9412494             0.15591627
      0.6834831             0.60768372
      0.2222899             0.39195739
''', optimize=True)}
)
print('num primitive GTOs = ', mol.npgto_nr())
