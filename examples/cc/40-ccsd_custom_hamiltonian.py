#!/usr/bin/env python

'''
Six-site 1D U/t=2 Hubbard-like model system with PBC at half filling.
The model is gapped at the mean-field level
'''

import numpy
from pyscf import gto, scf, ao2mo, cc

mol = gto.M(verbose=4)
n = 6
mol.nelectron = n
# Setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
# attribute) to be used in the post-HF calculations.  Without this parameter,
# some post-HF method (particularly in the MO integral transformation) may
# ignore the customized Hamiltonian if memory is not enough.
mol.incore_anyway = True

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


# In PySCF, the customized Hamiltonian needs to be created once in mf object.
# The Hamiltonian will be used everywhere whenever possible.  Here, the model
# Hamiltonian is passed to CCSD object via the mf object.

mycc = cc.RCCSD(mf)
mycc.kernel()
e,v = mycc.ipccsd(nroots=3)
print(e)

# Complex MO coefficients require the complex-capable restricted solver.
# Orbital phases leave this Hamiltonian and the resulting energy unchanged.
# Explicit rccsd.RCCSD also makes the implementation choice clear.
from pyscf.cc import rccsd
complex_mo = mf.mo_coeff.astype(complex) * numpy.exp(0.2j * numpy.arange(n))
# The complex in-core transformation needs the full AO integral tensor,
# rather than the symmetry-packed tensor used for the real calculation.
complex_mf = mf.copy()
complex_mf._eri = ao2mo.restore(1, mf._eri, n)
complex_cc = rccsd.RCCSD(complex_mf, mo_coeff=complex_mo).run()
print('Real and complex-orbital RCCSD total energies', mycc.e_tot, complex_cc.e_tot)
