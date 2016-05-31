#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Assign FCI wavefunction symmetry
'''

import numpy
from pyscf import gto, scf, fci

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='631g', symmetry=True)
m = scf.RHF(mol)
m.kernel()
norb = m.mo_energy.size
nelec = mol.nelectron

fs = fci.FCI(mol, m.mo_coeff)
fs.wfnsym = 'E1gx'
e, c = fs.kernel(verbose=5)
print('Energy of %s state %.12f' % (fs.wfnsym, e))
