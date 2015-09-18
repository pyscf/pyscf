#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto

'''
Input molecule geometry with Z-matrix format
'''

mol = gto.M(
    atom = gto.from_zmatrix('''
    C
    H    1  1.2
    H    1  1.2  2  109.5
    H-1  1  1.2  2  109.5  3  120
    H-2  1  1.2  2  109.5  3  -120
    '''),
)
