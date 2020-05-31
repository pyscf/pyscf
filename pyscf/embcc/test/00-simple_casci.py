#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CASCI calculation.
'''
import numpy as np
import pyscf

mol = pyscf.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'ccpvdz',
    verbose=4
    )

mf = mol.RHF()
mf.run()

# 6 orbitals, 8 electrons
from pyscf import mcscf
cas = mcscf.CASCI(mf, 6, 6)
Etot, Ecas, ci, mo_coeff, mo_energy = cas.kernel()


print(Etot)
print(Ecas)

print(np.allclose(mo_energy, mf.mo_energy))
print(np.allclose(mo_coeff, mf.mo_coeff))
#print(E)



