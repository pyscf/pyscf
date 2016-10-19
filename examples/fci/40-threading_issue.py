#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Using multithreading (OMP_NUM_THREADS=8 eg) causes numerical instability for
FCI solver.
'''

import numpy
from pyscf import gto, lo, fci, ao2mo, scf

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

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, max_cycle=500, max_space=100, verbose=5)
print(e + mol.energy_nuc())

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, ci0=ci, max_cycle=500, max_space=100, verbose=5)
print(e + mol.energy_nuc())

e, ci = fci.direct_spin0.kernel(h1, h2, 10, 10, ci0=ci, max_cycle=500, max_space=100, verbose=5)
print(e + mol.energy_nuc())

