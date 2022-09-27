#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Allow Sz value to be changed during the SCF iteration.
'''

from pyscf import gto, scf
mol = gto.M(atom='O 0 0 0; O 0 0 1')
mf = scf.UHF(mol)
mf.verbose = 4
mf = scf.addons.dynamic_sz_(mf)
mf.kernel()

