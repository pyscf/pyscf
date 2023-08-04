'''
K-point symmetry adapted KRCCSD
Reference result:
converged SCF energy = -6.97030826215306
E(KsymAdaptedRCCSD) = -7.08763343861385  E_corr = -0.1173251757710611
'''
import numpy as np
from pyscf.pbc import gto, df, scf, cc

a = 5.431020511
xyz = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) * a

atom = []
for ix in xyz:
    atom.append(['Si', list(ix)])

cell = gto.Cell()
cell.atom = atom
cell.basis = 'gth-dzv'
cell.pseudo  = 'gth-pade'
cell.a = np.array([[0., .5, .5],[.5, 0., .5],[.5, .5, 0.]]) * a
cell.max_memory = 16000
cell.verbose = 4
cell.space_group_symmetry = True
cell.build()

kmesh = [4,2,2]
kpts = cell.make_kpts(kmesh, with_gamma_point=True, space_group_symmetry=True)

gdf = df.RSGDF(cell, kpts=kpts.kpts)
gdf._cderi = gdf._cderi_to_save = 'si_dzv_rsgdf_4x2x2.h5'
gdf.build()

kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
kmf.with_df = gdf
kmf.kernel()

kcc = cc.KsymAdaptedRCCSD(kmf)
# setting eris_outcore to True enforces the integrals to be stored on the disk
#kcc.eris_outcore = True
# setting ktensor_direct to True computes the symmetry related blocks of tensors on-the-fly;
# this reduces the memory usage but may introduce more overhead for small k-point meshes.
kcc.ktensor_direct = True
kcc.kernel()
