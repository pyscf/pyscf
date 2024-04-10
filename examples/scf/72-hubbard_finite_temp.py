#!/usr/bin/env python
#
# Author: Chong Sun <sunchong137@gmail.com>
#

'''
Simulate model systems with HF.
Half-filled Hubbard model.
'''

from pyscf import gto, scf , ao2mo
import numpy

def _hubbard_hamilts_pbc(L, U):
    h1e = numpy.zeros((L, L))
    g2e = numpy.zeros((L,)*4)
    for i in range(L):
        h1e[i, (i+1)%L] = h1e[(i+1)%L, i] = -1 
        g2e[i, i, i, i] = U
    return h1e, g2e

L = 10
U = 4

mol = gto.M()
mol.nelectron = L
mol.nao = L
mol.spin = 0
mol.incore_anyway = True
mol.build()

# set hamiltonian
h1e, eri = _hubbard_hamilts_pbc(L, U)
mf = scf.UHF(mol)
mf.get_hcore = lambda *args: h1e
mf._eri = ao2mo.restore(1, eri, L)
mf.get_ovlp = lambda *args: numpy.eye(L)
mf.kernel()

# finite temperature 
from pyscf.scf import addons
beta = 1
mf_ft = addons.smearing(mf, sigma=1./beta, method='fermi', fix_spin=True)
mf_ft.kernel()
