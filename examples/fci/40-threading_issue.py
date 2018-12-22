#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
An example to set OMP threads in FCI calculations. In old pyscf versions,
different number of OpenMP threads may lead to slightly different answers.

This issue was fixed. see github issue #249.
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

#
# Another Example.
#
import h5py
with h5py.File('spin_op_hamiltonian.h5', 'r') as f:
    h1 = lib.unpack_tril(f['h1'].value)
    h2 = f['h2'].value

norb = 10
nelec = (5,5)
na = fci.cistring.num_strings(norb, nelec[0])
c0 = numpy.zeros((na,na))
c0[0,0] = 1
solver = fci.addons.fix_spin_(fci.direct_spin0.FCI())

# Smooth convergence was found with single thread.
solver.threads = 1
solver.kernel(h1, h2, norb, nelec, ci0=c0, verbose=5)

# When switching to multi-threads, numerical fluctuation leads to convergence
# problem
solver.threads = 4
solver.kernel(h1, h2, norb, nelec, ci0=c0, verbose=5)
