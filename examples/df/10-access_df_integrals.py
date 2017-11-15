#!/usr/bin/env python

'''
This example shows how to read the density-fitting 3-index tensor from the DF
calculations or directly from the DF module.
'''

import numpy
import h5py
from pyscf import gto, scf, df

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
  Fe  7.84036274632279      2.40948832380662      3.90857987198295
  S   8.11413397508734      3.34683967317511      5.92473122721237
  Cl  9.42237962692288      2.83901882053830      2.40523971787167
  N   7.05276738605521      2.42445066370842      8.66076404425459
'''
mol.basis = 'ccpvdz'
mol.build()

mf = scf.RHF(mol).density_fit()
mf.kernel()

#
# After calling denisty fitting method for SCF, MCSCF methods etc, the density
# fitting integral (3-index) tensor can be accessed from  mf.with_df._cderi
# _cderi is short for cholesky decomposed (CD) electron repulsion integral.
# _cderi tensor can be either numpy.ndarray or HDF5 file (key is 'j3c').
# _cderi shape is (row,col) = (CD-vector-index, compressed-AO-pair)
#
print(mf.with_df._cderi.shape)

#
# By default, the CD tensor will be destructed.  To save the CD tensor for
# future use, specify the filename in the attribute mf.with_df._cderi_to_save
#
mf = scf.RHF(mol).density_fit()
mf.with_df._cderi_to_save = 'saved_cderi.h5'
mf.kernel()

#
# To load the precomputed CD tensors in another calculation.
# See also example/df/40-precompute_df_integrals.py
#
# mf = scf.RHF(mol).density_fit()
# mf.with_df._cderi = '/path/to/saved/tensor/file'
# mf.kernel()

#
# _cderi can be generated in the DF object without the DF-SCF calculations
#
mydf = df.DF(mol)
mydf.auxbasis = df.make_auxbasis(mol)
mydf._cderi_to_save = 'saved_cderi.h5'
mydf.build()

#
# DF integral tensor can also be generated through the call to cholesky_eri
# function
#
cderi = df.incore.cholesky_eri(mol, auxbasis='weigend')
df.outcore.cholesky_eri(mol, 'saved_cderi.h5', dataname='j3c',
                        auxbasis=df.make_auxbasis(mol))

