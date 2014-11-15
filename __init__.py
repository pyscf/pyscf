'''
pyscf
=====
'''

__version__ = '0.7'

import os
import gto
import lib
import scf
import ao2mo
#import symm

# modules in ./future are in test
__path__.append(os.path.join(os.path.dirname(__file__), 'future'))
