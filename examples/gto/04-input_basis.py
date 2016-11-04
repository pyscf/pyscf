#!/usr/bin/env python

'''
This example covers seven different ways to input basis
1. One type of basis set for all atoms.
2. Different basis for different elements.
3. Different basis for same elements (different atoms).
4. Use gto.basis.parse and gto.basis.load functions to input user-specified basis.
5. Uncontracted basis.
6. Even tempered gaussian basis.
7. Internal format (not recommended)
'''

from pyscf import gto
import os

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
# Use gto.basis.parse and gto.basis.load functions to input basis
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'O': gto.parse('''
# Parse NWChem format basis string (see https://bse.pnl.gov/bse/portal).
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
             'H1': gto.load(basis_file_from_user, 'H'),
             'H2': gto.load('sto-3g', 'He')  # or use basis of another atom
            }
)

#
# Uncontracted basis
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
# Even tempered gaussian basis
#
mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'H': 'sto3g',
# even-temper gaussians alpha*beta^i, where i = 0,..,n
#                                  (l, n, alpha, beta)
             'O': gto.expand_etbs([(0, 4, 1.5, 2.2),  # s-function
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
#
# Functions gto.basis.load and got.basis.parse  convert the input to the
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
print(mol.basis)
