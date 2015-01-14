'''
pyscf
=====
'''

__version__ = '0.10'

import os
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import ao2mo
#import symm

# modules in ./future are in test
__path__.append(os.path.join(os.path.dirname(__file__), 'future'))
