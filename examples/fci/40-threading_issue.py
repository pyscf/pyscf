#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Different number of OpenMP threads may lead to slightly different answers
'''

from functools import reduce
import numpy
from pyscf import gto, lo, fci, ao2mo, scf, lib

mol = gto.M(atom=[('H', 0, 0, i*1.8) for i in range(10)],
            basis = 'sto6g', unit='B')
s = mol.intor('cint1e_ovlp_sph')
orb = lo.lowdin(s)
#mf = scf.RHF(mol).run()
#orb = mf.mo_coeff

h1 = mol.intor('cint1e_nuc_sph')
h1+= mol.intor('cint1e_kin_sph')
h1 = reduce(numpy.dot, (orb.T, h1, orb))
h2 = ao2mo.kernel(mol, orb)

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, ecore=mol.energy_nuc(),
                                max_cycle=500, max_space=100, verbose=5)
print(e)

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, ecore=mol.energy_nuc(), ci0=ci,
                                max_cycle=500, max_space=100, verbose=5)
print(e)

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, ecore=mol.energy_nuc(), ci0=ci,
                                max_cycle=500, max_space=100, verbose=5)
print(e)


#
# Reducing OMP threads can improve the numerical stability
#

# Set OMP_NUM_THREADS to 1
lib.num_threads(1)

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, ecore=mol.energy_nuc(),
                                max_cycle=500, max_space=100, verbose=5)
print(e)

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, ecore=mol.energy_nuc(), ci0=ci,
                                max_cycle=500, max_space=100, verbose=5)
print(e)
