#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to create a density fitting calculation.

See also
examples/scf/20-density_fitting.py
examples/pbc/11-gamma_point_all_electron_scf.py
'''

from pyscf import gto, scf, df
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc import df as pdf

#
# Method 1: Initialize a DF-SCF object based on the given SCF object.
#
# An attribute .with_df is created in the DF-SCF object.  with_df attribute
# holds all methods and informations of the DF integrals.
#
mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = df.density_fit(scf.RHF(mol), auxbasis='weigend')
print(mf.with_df)
mf.kernel()

#
# Method 2: SCF object has density_fit method.
#
mf = scf.RHF(mol).density_fit(auxbasis='weigend')
mf.kernel()

# DF method can be switched off by assigning None to with_df object
mf.with_df = None
mf.kernel()


#
# In PBC calculations, DF method can be used for all-electron calculation.
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
cell.verbose = 4
cell.build()
kpts = cell.make_kpts([2,2,2])

mf = pdft.KRKS(cell, kpts=kpts).density_fit(auxbasis='ahlrichs')
mf.kernel()
