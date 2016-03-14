#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import h5py
from pyscf import gto, df, scf, mcscf

'''
Input Cholesky decomposed integrals for CASSCF
'''

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')

#
# Integrals in memory
#
int3c = df.incore.cholesky_eri(mol, auxbasis='ccpvdz-fit')
mf = scf.density_fit(scf.RHF(mol))
mf._cderi = int3c
mf.kernel()

# 3-cetner DF or Cholesky decomposed integrals need to be initialized once in
# mf._cderi.  DFCASSCF method automatically use the approximate integrals
mc = mcscf.DFCASSCF(mf, 8, 8)
mc.kernel()


#
# Integrals on disk
#
ftmp = tempfile.NamedTemporaryFile()
df.outcore.cholesky_eri(mol, ftmp.name, auxbasis='ccpvdz-fit')

with h5py.File(ftmp.name, 'r') as file1:
    mf = scf.density_fit(scf.RHF(mol))
# Note, here the integral object file1['eri_mo'] are not loaded in memory.
# It is still the HDF5 array object held on disk.  The HDF5 array can be used
# the same way as the regular numpy ndarray stored in memory.
    mf._cderi = file1['eri_mo']
    mf.kernel()

# Note the mc object must be put inside the "with" statement block because it
# still needs access the HDF5 integral array on disk
    mc = mcscf.DFCASSCF(mf, 8, 8)
    mc.kernel()

