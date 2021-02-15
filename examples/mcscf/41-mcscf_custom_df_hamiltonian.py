#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import h5py
from pyscf import gto, df, scf, mcscf

'''
Using the Cholesky decomposed 2-electron integrals to define the Hamiltonian in CASSCF

See also examples/df/40-precompute_df_ints.py
'''

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')

#
# Integrals in memory. The size of the integral array is (M,N*(N+1)/2), where
# the last two AO indices are compressed due to the symmetry
#
int3c = df.incore.cholesky_eri(mol, auxbasis='ccpvdz-fit')
mf = scf.density_fit(scf.RHF(mol))
mf.with_df._cderi = int3c
mf.kernel()

# 3-cetner DF or Cholesky decomposed integrals need to be initialized once in
# mf.with_df._cderi.  DFCASSCF method automatically use the approximate integrals
mc = mcscf.DFCASSCF(mf, 8, 8)
mc.kernel()


#
# Integrals on disk
#
ftmp = tempfile.NamedTemporaryFile()
df.outcore.cholesky_eri(mol, ftmp.name, auxbasis='ccpvdz-fit')

with h5py.File(ftmp.name, 'r') as file1:
    mf = scf.density_fit(scf.RHF(mol))
# Note, here the integral object file1 are not loaded in memory.
# It is still the HDF5 array object held on disk.  The HDF5 array can be used
# the same way as the regular numpy ndarray stored in memory.
    mf.with_df._cderi = file1
    mf.kernel()

# Note the mc object must be put inside the "with" statement block because it
# still needs access the HDF5 integral array on disk
    mc = mcscf.DFCASSCF(mf, 8, 8)
    mc.kernel()

