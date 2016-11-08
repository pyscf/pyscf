#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generally, the CASSCF solver does NOT return the natural orbitals.
'''

import numpy
from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)
myhf = scf.RHF(mol).run()

# 6 orbitals, 8 electrons
mycas = mcscf.CASSCF(myhf, 6, 8)

#
# Freeze the inner most two 1s orbitals
#
mycas.frozen = 2
mycas.kernel()

#
# Freeze orbitals based on the list of indices.  Two HF core orbitals and two HF
# virtual orbitals are excluded from CASSCF optimization.
#
mycas.frozen = [0,1,26,27]
mycas.kernel()

#
# Partially freeze the active space so that the frozen orbitals are always in
# the active space.  It can help CASSCF converge to reasonable solution.
#
mycas.frozen = [5,6,7,8]
mycas.kernel()

