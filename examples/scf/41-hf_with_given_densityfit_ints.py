#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Input Cholesky decomposed integrals for SCF module by overwriting the _cderi attribute.

See also:
examples/df/40-precompute_df_ints.py
'''

import tempfile
import h5py
from pyscf import gto, df, scf

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')

# Integrals in memory
int3c = df.incore.cholesky_eri(mol, auxbasis='ccpvdz-fit')

# Integrals on disk
ftmp = tempfile.NamedTemporaryFile()
df.outcore.cholesky_eri(mol, ftmp.name, auxbasis='ccpvdz-fit')


fake_mol = gto.M()
fake_mol.nelectron = 10  # Note: you need define the problem size

#
# Redefine the 2-electron integrals by overwriting mf._cderi with the given
# 3-center intagrals.  You need create density-fitting SCF object.  The
# regulare SCF object cannot hold the 3-center integrals.
#
mf = scf.density_fit(scf.RHF(fake_mol))
mf.get_hcore = lambda *args: (mol.intor('cint1e_kin_sph') +
                              mol.intor('cint1e_nuc_sph'))
mf.get_ovlp = lambda *args: mol.intor('cint1e_ovlp_sph')
mf._cderi = int3c
mf.init_guess = '1e'  # Initial guess from Hcore
mf.kernel()


#
# Assuming the 3-center integrals happens too huge to be held in memory, there
# is a hacky way to input the integrals.  The h5py dataset can be accessed in
# the same way as the numpy ndarray.
#
with h5py.File(ftmp.name, 'r') as file1:
    mf = scf.density_fit(scf.RHF(fake_mol))
    mf._cderi = file1['j3c']
    mf.get_hcore = lambda *args: (mol.intor('cint1e_kin_sph') +
                                  mol.intor('cint1e_nuc_sph'))
    mf.get_ovlp = lambda *args: mol.intor('cint1e_ovlp_sph')
    mf.init_guess = '1e'  # Initial guess from Hcore
    mf.kernel()

