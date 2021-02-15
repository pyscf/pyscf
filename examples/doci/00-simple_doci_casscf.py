#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run DOCI-CASCI and DOCI-CASSCF calculation.
'''

from pyscf import gto
from pyscf import doci

mol = gto.M(atom='N 0 0 0; N 0 0 2.', basis='6-31g')
mf = mol.RHF().run()
mc = doci.CASSCF(mf, 18, 14)
mc.verbose = 4
mc.kernel()

mc = doci.CASCI(mf, 18, 14)
mc.kernel()
