'''
RPA with k-points sampling
'''

from pyscf.pbc import gto, df, dft, scf
from pyscf.pbc.gw.krpa import KRPA
from pyscf.pbc.gw.kurpa import KURPA

# spin-restricted RPA
cell = gto.Cell()
cell.build(
    unit='angstrom',
    a="""
            0.000000     1.783500     1.783500
            1.783500     0.000000     1.783500
            1.783500     1.783500     0.000000
        """,
    atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
    dimension=3,
    max_memory=12000,
    verbose=5,
    pseudo='gth-pbe',
    basis='gth-dzv',
    precision=1e-12,
)

kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
gdf = df.RSGDF(cell, kpts)
gdf.build()
kmf = scf.KRHF(cell, kpts).rs_density_fit()
kmf.with_df = gdf
kmf.kernel()

# RPA with finite-size correction
rpa = KRPA(kmf)
rpa.fc = True
rpa.kernel()
# RPA with finite-size correction
rpa = KRPA(kmf)
rpa.fc = False
rpa.kernel()
# low-memory routine
rpa = KRPA(kmf)
rpa.outcore = True
rpa.segsize = 2
rpa.kernel()

# Na (metallic)
cell = gto.Cell()
cell.build(
    unit='angstrom',
    a="""
         -2.11250000000000   2.11250000000000   2.11250000000000
        2.11250000000000  -2.11250000000000   2.11250000000000
        2.11250000000000   2.11250000000000  -2.11250000000000
        """,
    atom="""Na   0.00000   0.00000   0.00000""",
    dimension=3,
    max_memory=126000,
    verbose=5,
    pseudo='gth-pade',
    basis='gth-dzvp-molopt-sr',
    precision=1e-10,
)

kpts = cell.make_kpts([2, 2, 1], scaled_center=[0, 0, 0])
gdf = df.RSGDF(cell, kpts)
gdf.build()

kmf = dft.KRKS(cell, kpts).rs_density_fit()
kmf = scf.addons.smearing_(kmf, sigma=5e-3, method='fermi')
kmf.xc = 'lda'
kmf.with_df = gdf
kmf.kernel()

rpa = KRPA(kmf)
rpa.kernel()
# use ACFDT exchange energy
rpa = KRPA(kmf)
rpa.acfd_exx = True
rpa.kernel()

# spin-unrestricted RPA
cell = gto.Cell()
cell.build(
    unit='B',
    a=[[0.0, 6.74027466, 6.74027466], [6.74027466, 0.0, 6.74027466], [6.74027466, 6.74027466, 0.0]],
    atom="""H 0 0 0
            H 1.68506866 1.68506866 1.68506866
            H 3.37013733 3.37013733 3.37013733""",
    basis='gth-dzvp',
    pseudo='gth-pade',
    verbose=5,
    charge=0,
    spin=1,
)

cell.spin = cell.spin * 3
kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
gdf = df.RSDF(cell, kpts)
gdf.build()

kmf = scf.KUHF(cell, kpts, exxdiv='ewald').rs_density_fit()
kmf = scf.addons.smearing_(kmf, sigma=5e-3, method='fermi')
kmf.xc = 'lda'
kmf.with_df = gdf
kmf.kernel()

rpa = KURPA(kmf)
rpa.kernel()
# use ACFDT exchange energy
rpa = KURPA(kmf)
rpa.acfd_exx = True
rpa.kernel()
