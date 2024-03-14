#!/usr/bin/env python

'''
This example shows how to use a pre-computed GDF tensor in a PBC calculation
'''

from pyscf.pbc import gto

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.basis = 'sto3g'
cell.unit = 'B'
cell.verbose = 5
cell.build()

kpts = cell.make_kpts([2,2,2])

# First, generate GDF tensor and saved in a file. Filename needs to be assigned
# to the attribute _cderi_to_save
mf = cell.KRHF(kpts=kpts).density_fit()
mf.with_df._cderi_to_save = 'gdf-sample.h5'
mf.with_df.build()

# To reuse the pre-computed GDF CDERI tensor, assign the GDF CDERI file to
# the attribute _cderi
mf = cell.KRHF(kpts=kpts).density_fit()
mf.with_df._cderi = 'gdf-sample.h5'
mf.run()

# For DFT hybrid functionals, flag _j_only needs to be set manually.
# DFT program assumes that the GDF CDERI file does not contain integral data for
# HF exchange matrix. Without specifying this flag, GDF cderi tensor may be
# regenerated for hybrid functionals.
mf = cell.KRKS(kpts=kpts).density_fit()
mf.xc = 'pbe'
mf.with_df._cderi = 'gdf-sample.h5'
mf.with_df._j_only = False
mf.run()
