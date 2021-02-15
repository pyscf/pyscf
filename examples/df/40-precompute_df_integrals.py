#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to save the DF integrals in one calculation and use the
integrals in another one.  This can be particularly useful in all-electron PBC
calculations.  The time-consuming DF integral generation can be done once and
reused many times.
'''

import tempfile
from pyscf import gto, scf, df
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft

tmpf = tempfile.NamedTemporaryFile()
file_to_save_df_ints = tmpf.name
print('DF integral is saved in %s' % file_to_save_df_ints)

#
# Save the density fitting integrals in a file.
# After calling .density_fit() function for SCF object, attribute .with_df
# will be created for the SCF object.  If with_df._cderi_to_save is specified,
# the DF integrals will be saved in this file.
#
mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol).density_fit(auxbasis='weigend')
mf.with_df._cderi_to_save = file_to_save_df_ints
mf.kernel()

#
# The DF integral file can be used in an separated calculation.
# Attribute with_df._cderi is the filename which saves the DF integrals.  This
# is a read-only file, integrals in this file will not be modified in the
# You can feed integrals using ._cderi (HDF5) file from any source/program or
# an early calculation.
#
mf = scf.RHF(mol).density_fit()
mf.with_df._cderi = file_to_save_df_ints
mf.kernel()

#
# Attribute with_df._cderi also can be a Numpy array.
# In the following example, df.incore.cholesky_eri function generates the DF
# integral tensor in a numpy array.
#
int3c = df.incore.cholesky_eri(mol, auxbasis='ahlrichs')
mf = scf.RHF(mol).density_fit()
mf.with_df._cderi = int3c
mf.kernel()


#
# Same mechanism can be used in PBC system.  Generating DF integrals under PBC
# is very expensive.  These DF integrals can be generated once in a file and
# reused later in another calculation.
#

cell = pgto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = '6-31g'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.mesh = [10]*3
#cell.verbose = 4
cell.build()

# Using density fitting for all-electron calculation
mf = pdft.RKS(cell).density_fit(auxbasis='ahlrichs')
# Saving DF integrals in file
mf.with_df._cderi_to_save = file_to_save_df_ints
mf.kernel()

# Using DF integrals in a separated calculation by specifying with_df._cderi
# attribute.
mf.with_df._cderi = file_to_save_df_ints
mf.kernel()
