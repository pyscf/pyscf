#!/usr/bin/env python

'''
G0W0 with k-points sampling
'''
import numpy as np
from pyscf.pbc import df, gto, scf, gw

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

kpts = cell.make_kpts([2, 2, 2])
gdf = df.RSDF(cell, kpts)
gdf.build()

# restricted KGW
kmf = scf.KRKS(cell, kpts).rs_density_fit()
kmf.with_df = gdf
kmf.kernel()

# KRGWAC using analytical continuation
mygw = gw.krgw_ac.KRGWAC(kmf)
mygw.kernel()

# KRGWAC low-memory routine
# finite-size correction is not implemented for outcore routine
mygw = gw.krgw_ac.KRGWAC(kmf)
mygw.outcore = True
mygw.fc = False
mygw.kernel()

# KRGWAC full self-energy and density of states
mygw = gw.krgw_ac.KRGWAC(kmf)
mygw.fullsigma = True
mygw.kernel()
omega = np.linspace(-1, 1, 201)
# gf: GW Green's function; gf0: DFT Green's function; sigma: self-energy
gf, gf0, sigma = mygw.make_gf(omega, eta=1e-2)
print("k=0 density of states")
for i in range(len(omega)):
    print(omega[i], -np.trace(gf[0, :, :, i].imag) / np.pi)

# With CD frequency integration
#mygw = gw.KRGW(kmf, freq_int='cd')
#mygw.kernel()
#print("KRGW-CD energies =", mygw.mo_energy)

# restricted KGW
kmf = scf.KUKS(cell, kpts).rs_density_fit()
kmf.with_df = gdf
kmf.kernel()

# KUGWAC using analytical continuation with finite-size correction
mygw = gw.kugw_ac.KUGWAC(kmf)
mygw.fc = True
mygw.kernel()

# KUGWAC full self-energy and density of states
mygw = gw.kugw_ac.KUGWAC(kmf)
mygw.fullsigma = True
mygw.kernel()
omega = np.linspace(-1, 1, 201)
# gf: GW Green's function; gf0: DFT Green's function; sigma: self-energy
gf, gf0, sigma = mygw.make_gf(omega, eta=1e-2)
print("k=0 density of states: alpha beta")
for i in range(len(omega)):
    print(omega[i], -np.trace(gf[0, 0, :, :, i].imag) / np.pi, -np.trace(gf[0, 1, :, :, i].imag) / np.pi)
