#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import h5py
from pyscf import gto, df, scf

'''
Input Cholesky decomposed integrals for SCF module by overwriting the _cderi attribute.
'''

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')

# Integrals in memory
int3c = df.incore.cholesky_eri(mol, auxbasis='ccpvdz-fit')
mf = scf.density_fit(scf.RHF(mol))
mf._cderi = int3c
mf.kernel()


# Integrals on disk
ftmp = tempfile.NamedTemporaryFile()
df.outcore.cholesky_eri(mol, ftmp.name, auxbasis='ccpvdz-fit')
with h5py.File(ftmp.name, 'r') as file1:
    mf = scf.density_fit(scf.RHF(mol))
    mf._cderi = file1['eri_mo']
    mf.kernel()
