#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf, ao2mo

'''
Customizing Hamiltonian for SCF module.

Three steps to define Hamiltonian for SCF:
1. Specify the number of electrons. (Note mole object must be "built" before doing this step)
2. Overwrite three attributes of scf object
    .get_hcore
    .get_ovlp
    ._eri
3. Specify initial guess (to overwrite the default atomic density initial guess)

Note you will see warning message on the screen:

        Overwritten attributes  get_ovlp get_hcore  of <class 'pyscf.scf.hf.RHF'>

'''

mol = gto.M()
n = 10
mol.nelectron = n

mf = scf.RHF(mol)
h1 = numpy.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = 4.0

mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
# ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
# ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
mf._eri = ao2mo.restore(8, eri, n)

mf.kernel()

# If you need to run post-HF calculations based on the customized Hamiltonian,
# setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
# attribute) to be used.  Without this parameter, some post-HF method
# (particularly in the MO integral transformation) may ignore the customized
# Hamiltonian if memory is not enough.
mol.incore_anyway = True
