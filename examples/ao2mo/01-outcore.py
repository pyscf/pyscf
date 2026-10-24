#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import h5py
import numpy
from pyscf import gto, scf, ao2mo
from pyscf import lib
from pyscf.ao2mo import semi_incore

'''
Save the transformed integrals in the given file in HDF5 format
'''

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
)

myhf = scf.RHF(mol)
myhf.kernel()

orb = myhf.mo_coeff
ftmp = lib.NamedTemporaryFile()
print('MO integrals are saved in file  %s  under dataset "eri_mo"' % ftmp.name)
ao2mo.kernel(mol, orb, ftmp.name)

# Read integrals via h5py module
with h5py.File(ftmp.name) as f:
    eri_4fold = f['eri_mo']
    print('MO integrals (ij|kl) with 4-fold symmetry i>=j, k>=l have shape %s' %
          str(eri_4fold.shape))

# Request all four MO indices explicitly when the compressed pair layout is
# unsuitable for subsequent tensor operations.
ao2mo.outcore.general(mol, (orb,) * 4, ftmp.name, dataname='eri_full',
                      compact=False)
with h5py.File(ftmp.name) as f:
    nmo = orb.shape[1]
    eri_full = f['eri_full'][:].reshape((nmo,) * 4)
    print('Uncompressed MO integral shape:', eri_full.shape)

# Semi-incore transformation starts from an AO integral array and can write
# just the occupied-first block to disk.
occ = orb[:, myhf.mo_occ > 0]
semi_incore.general(myhf._eri, (occ, orb, orb, orb), ftmp.name,
                    dataname='eri_occ')
with h5py.File(ftmp.name) as f:
    eri_occ = f['eri_occ'][:]
    eri_ref = ao2mo.incore.general(myhf._eri, (occ, orb, orb, orb))
    print('Semi-incore occupied block agrees:', numpy.allclose(eri_occ, eri_ref))

# The same interface also writes a complete MO integral matrix.
semi_incore.general(myhf._eri, (orb,) * 4, ftmp.name, dataname='eri_all')
with h5py.File(ftmp.name) as f:
    print('Semi-incore full transform agrees:',
          numpy.allclose(f['eri_all'][:], ao2mo.incore.full(myhf._eri, orb)))
