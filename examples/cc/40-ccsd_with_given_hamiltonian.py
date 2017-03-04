#!/usr/bin/env python
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import cc

#
# 1D anti-PBC Hubbard model at half filling
#
mol = gto.M(verbose=4)
mol.nelectron = 6
n = 12

h1 = numpy.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[n-1,0] = h1[0,n-1] = -1.0
eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = 2.0

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
mf._eri = ao2mo.restore(8, eri, n)
mf.kernel()

#
# In PySCF, the faked Hamiltonians just need to be created once in mf object,
# and can be used with mf object everywhere.  Here, the Hubbard model is
# passed to CCSD object with the mf object.
#
mycc = cc.CCSD(mf)
mycc.kernel()

