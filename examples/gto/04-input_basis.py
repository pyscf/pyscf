#!/usr/bin/env python
from pyscf import gto
import os

'''
Use gto.basis.parse and gto.basis.load functions to input user-specified basis functions.
'''

dirnow = os.path.realpath(os.path.join(__file__, '..'))
basis_file_from_user = os.path.join(dirnow, 'h_sto3g.dat')

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

mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'O': 'unc-ccpvdz', # prefix "unc-" will uncontract the ccpvdz basis.
                                # It is equivalent to assigning
                                #   'O': gto.uncontract(gto.load('ccpvdz', 'O')),
             'H': 'ccpvdz'  # H1 H2 will use the same basis ccpvdz
            }
)


mol = gto.M(
    atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
    basis = {'H': 'sto3g',
# even-temper gaussians alpha*beta^i, where i = 0,..,n
#                                  (l, n, alpha, beta)
             'O': gto.expand_etbs([(0, 4, 1.5, 2.2),  # s-function
                                   (1, 2, 0.5, 2.2)]) # p-function
            }
)

